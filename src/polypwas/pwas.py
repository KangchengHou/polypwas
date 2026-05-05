"""PWAS computation: z-scores, co-regulation matrices, and variance explained."""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union

from .ld import BlockLDM, read_ldm
from .store import BlockWgt
from .utils import MultiTableReader, get_exon_mask


def sumstats_pwas(
    ldm_dir: str,
    weights: Union[pd.DataFrame, pd.Series],
    gwasz: pd.Series,
    verbose: bool = False,
) -> Union[pd.Series, float]:
    """Compute PWAS Z-score: Z = (w' @ z) / sqrt(w' @ R @ w).

    Uses full (non-block) LD matrices.

    Parameters
    ----------
    ldm_dir : str
        Path to the SBayesRC LD matrix directory.
    weights : pd.DataFrame or pd.Series
        Protein weights per SNP. If DataFrame, each column is a protein.
    gwasz : pd.Series
        GWAS Z-scores per SNP.
    verbose : bool
        Show progress bar.

    Returns
    -------
    pd.Series or float
        PWAS Z-scores (one per protein, or scalar if weights is a Series).
    """
    is_series = isinstance(weights, pd.Series)
    if is_series:
        weights = weights.to_frame()

    ldm_info = pd.read_csv(f"{ldm_dir}/ldm.info", sep="\t", index_col="Block")
    snp_info = pd.read_csv(f"{ldm_dir}/snp.info", sep="\t", index_col="ID")
    assert np.array_equal(snp_info.index, gwasz.index)
    assert np.array_equal(snp_info.index, weights.index)

    numer = 0
    denom = 0
    for idx in tqdm(ldm_info.index, desc="Computing PWAS Z across blocks", disable=not verbose):
        block_ldm, block_info = read_ldm(f"{ldm_dir}/b{idx}.ldm.full.bin")
        block_gwas = gwasz.loc[block_ldm.index]
        block_wgts = weights.loc[block_ldm.index, :]
        numer += block_wgts.T @ block_gwas
        denom += pd.Series(
            {col: block_wgts[col] @ block_ldm @ block_wgts[col] for col in block_wgts.columns}
        )

    z = numer / np.sqrt(denom)
    return z.item() if is_series else z


def load_hotspot_snps(pqtl_count_path, quantile=0.001):
    """Load pQTL hotspot SNPs (top `quantile` fraction by total_count).

    Parameters
    ----------
    pqtl_count_path : str or Path
        Path to ``{dataset}.pqtl_count.tsv``.
    quantile : float
        Fraction of SNPs to flag as hotspots (default 0.001 = top 0.1%).

    Returns
    -------
    set
        SNP IDs of hotspot SNPs.
    """
    df = pd.read_csv(pqtl_count_path, sep="\t", index_col=0)
    threshold = df["total_count"].quantile(1 - quantile)
    hotspots = set(df.index[df["total_count"] > threshold])
    return hotspots


def load_exclude_snps(snp_list_path):
    """Load a SNP exclusion list from a text file (one SNP ID per line).

    Parameters
    ----------
    snp_list_path : str or Path
        Path to a text file with one SNP ID per line.

    Returns
    -------
    set
        SNP IDs to exclude.
    """
    with open(snp_list_path) as f:
        return set(line.strip() for line in f if line.strip())


def _build_cis_trans_masks(block_info, protein_annot, pids, cis_window, subsets,
                           exclude_snps=None):
    """Build boolean mask arrays for cis/trans/genic/exonic subsets within a block.

    Vectorized: broadcasts SNP positions against protein coordinates using numpy,
    avoiding per-protein Python loops for the common cis/trans case.

    Parameters
    ----------
    exclude_snps : dict[str, set] or None
        Mapping from suffix to SNP ID sets for exclusion masks. For each key
        ``suffix``, generates ``cis_{suffix}`` and ``trans_{suffix}`` masks
        that zero out SNPs in the exclusion set.
        Example: ``{"nohotspot": hotspot_set, "noblood": blood_set}``
    """
    if exclude_snps is None:
        exclude_snps = {}

    n_snp = len(block_info)
    n_prot = len(pids)
    subsets_set = set(subsets)
    mask_dict = {}

    snp_chrom = block_info["Chrom"].values
    snp_pos = block_info["PhysPos"].values

    annot = protein_annot.loc[pids, :]
    prot_chrom = annot["CHROM"].values  # (n_prot,)
    prot_start = annot["START"].values  # (n_prot,)
    prot_end = annot["END"].values      # (n_prot,)

    # Collect all exclusion-derived subset names
    exclude_masks = set()
    for suffix in exclude_snps:
        exclude_masks.add(f"cis_{suffix}")
        exclude_masks.add(f"trans_{suffix}")

    # Shared intermediates: compute once when needed by any subset
    base_masks = {"cis", "trans", "cisgenic", "cisnongenic", "cisexonic", "cisnonexonic",
                  "cisproximal", "cisnonproximal"}
    all_sub_masks = base_masks | exclude_masks
    need_chrom_match = bool(subsets_set & all_sub_masks)
    need_cis = bool(subsets_set & ({"cis", "trans", "cisnongenic", "cisnonexonic", "cisnonproximal"}
                                   | exclude_masks))

    chrom_match = None
    cis = None

    if need_chrom_match:
        chrom_match = snp_chrom[:, None] == prot_chrom[None, :]

    if need_cis:
        cis = chrom_match & (snp_pos[:, None] >= (prot_start - cis_window)[None, :]) & (
            snp_pos[:, None] <= (prot_end + cis_window)[None, :]
        )
        if "cis" in subsets_set:
            mask_dict["cis"] = cis.astype(np.float32)
        if "trans" in subsets_set:
            mask_dict["trans"] = (~cis).astype(np.float32)

    # Exclusion-filtered masks
    for suffix, snp_set in exclude_snps.items():
        cis_key = f"cis_{suffix}"
        trans_key = f"trans_{suffix}"
        if subsets_set & {cis_key, trans_key}:
            not_excluded = ~np.isin(block_info.index.values, list(snp_set))  # (n_snp,)
            not_excluded_2d = not_excluded[:, None]  # broadcast to (n_snp, n_prot)
            if cis_key in subsets_set:
                mask_dict[cis_key] = (cis & not_excluded_2d).astype(np.float32)
            if trans_key in subsets_set:
                mask_dict[trans_key] = (~cis & not_excluded_2d).astype(np.float32)

    if subsets_set & {"cisgenic", "cisnongenic"}:
        cisgenic = chrom_match & (snp_pos[:, None] >= prot_start[None, :]) & (
            snp_pos[:, None] <= prot_end[None, :]
        )
        if "cisgenic" in subsets_set:
            mask_dict["cisgenic"] = cisgenic.astype(np.float32)
        if "cisnongenic" in subsets_set:
            mask_dict["cisnongenic"] = (cis & ~cisgenic).astype(np.float32)

    if subsets_set & {"cisproximal", "cisnonproximal"}:
        proximal_window = 2000  # 2kb flanking
        cisproximal = chrom_match & (
            snp_pos[:, None] >= (prot_start - proximal_window)[None, :]
        ) & (
            snp_pos[:, None] <= (prot_end + proximal_window)[None, :]
        )
        if "cisproximal" in subsets_set:
            mask_dict["cisproximal"] = cisproximal.astype(np.float32)
        if "cisnonproximal" in subsets_set:
            mask_dict["cisnonproximal"] = (cis & ~cisproximal).astype(np.float32)

    if subsets_set & {"cisexonic", "cisnonexonic"}:
        # Exonic requires per-protein loop (variable-length exon intervals)
        cisexonic = np.zeros((n_snp, n_prot), dtype=bool)
        for i, pid in enumerate(pids):
            exons = annot.loc[pid, "EXONS"]
            if chrom_match[:, i].any():
                cisexonic[:, i] = chrom_match[:, i] & get_exon_mask(exons, snp_pos)
        if "cisexonic" in subsets_set:
            mask_dict["cisexonic"] = cisexonic.astype(np.float32)
        if "cisnonexonic" in subsets_set:
            mask_dict["cisnonexonic"] = (cis & ~cisexonic).astype(np.float32)

    return mask_dict


def compute_pwas(
    wgt_paths: dict[str, str],
    gwas_paths: dict[str, str],
    ldm_dir: str,
    protein_annot: pd.DataFrame,
    wgt_index_col: str = None,
    cis_window=1e6,
    subsets=("cis", "cisgenic", "cisnongenic", "cisexonic", "cisnonexonic", "trans"),
):
    """Compute PWAS covariances (numerator of Z-scores) for multiple traits and proteins.

    Uses block-eigendecomposition LD and streaming weight reading for memory efficiency.

    Parameters
    ----------
    wgt_paths : dict[str, str]
        {protein_id: path_to_weights} mapping.
    gwas_paths : dict[str, str]
        {trait: path_to_gwas_sumstats} mapping.
    ldm_dir : str
        Path to LD matrix directory.
    protein_annot : pd.DataFrame
        Protein annotation with columns CHROM, START, END, EXONS, indexed by protein ID.
    wgt_index_col : str, optional
        Column name in weight files to use as index.
    cis_window : float
        Cis window in bp (default 1e6).
    subsets : tuple of str
        SNP subsets to compute separately.

    Returns
    -------
    pd.DataFrame
        Long-format with columns: TRAIT, ID, {SUBSET}_COV for each subset.
    """
    ldm = BlockLDM(ldm_dir)
    snp_info = pd.read_csv(f"{ldm_dir}/snp.info", sep="\t", index_col="ID")
    wgt_reader = MultiTableReader(wgt_paths, index_col=wgt_index_col, value_col="BETA")

    gwas_df = {}
    for trait, gwas_path in tqdm(gwas_paths.items(), desc="Reading GWAS summary statistics"):
        df = pd.read_csv(gwas_path, index_col="SNP", sep="\t").loc[snp_info.index, :]
        gwas_df[trait] = df["b"] / df["se"]
    gwas_df = pd.DataFrame(gwas_df)

    subsets = list(subsets)
    result_dict = {(mask, "numer"): 0.0 for mask in subsets}

    for block_idx, block_info in tqdm(snp_info.groupby("Block")):
        nsnp = len(block_info)
        wgt_mat = wgt_reader.next(nsnp)
        if wgt_index_col is None:
            wgt_mat.index = block_info.index

        gwas_mat = gwas_df.loc[block_info.index, :]
        assert np.array_equal(wgt_mat.index, block_info.index)
        assert np.array_equal(gwas_mat.index, block_info.index)

        freq = block_info["A1Freq"].values
        wgt_mat *= np.sqrt(2 * freq * (1 - freq))[:, None]

        mask_dict = _build_cis_trans_masks(
            block_info, protein_annot, wgt_mat.columns, cis_window, subsets
        )

        for mask in subsets:
            result_dict[(mask, "numer")] += (wgt_mat * mask_dict[mask]).T @ gwas_mat

    # Reshape to long format
    cov_dfs = {
        mask: result_dict[(mask, "numer")]
        .reset_index(names="ID")
        .melt(id_vars="ID", var_name="TRAIT", value_name=f"{mask.upper()}_COV")
        for mask in subsets
    }
    long_df = cov_dfs[subsets[0]]
    for mask in subsets[1:]:
        long_df = pd.merge(long_df, cov_dfs[mask], on=["TRAIT", "ID"])
    long_df = long_df[["TRAIT", "ID", *[f"{mask.upper()}_COV" for mask in subsets]]]
    return long_df


def compute_pwas_z(
    weights: Union[pd.DataFrame, BlockWgt],
    gwas_z: pd.Series,
    ldm: BlockLDM,
    gene_info: pd.DataFrame,
    cis_window: float = 1e6,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute cis/trans PWAS Z-scores for many proteins against one GWAS.

    Parameters
    ----------
    weights : pd.DataFrame or BlockWgt
        Per-protein SNP weights. DataFrame must be indexed by SNP ID with one
        column per protein. BlockWgt streams block by block.
    gwas_z : pd.Series
        GWAS Z-scores indexed by SNP ID, aligned to ``ldm.snp_info``.
    ldm : BlockLDM
        Block-eigen LD matrix.
    gene_info : pd.DataFrame
        Indexed by protein ID with columns CHROM, START, END.
    cis_window : float
        Cis window in bp.

    Returns
    -------
    pd.DataFrame with columns ID, CIS_Z, TRANS_Z.
    """
    snp_info = ldm.snp_info
    pids = list(gene_info.index)

    chrom_dtype = snp_info["Chrom"].dtype
    if pd.api.types.is_numeric_dtype(chrom_dtype):
        prot_chrom = pd.to_numeric(gene_info["CHROM"], errors="coerce").astype(chrom_dtype).values
    else:
        prot_chrom = gene_info["CHROM"].astype(str).values
    prot_start = gene_info["START"].values
    prot_end = gene_info["END"].values

    is_blockwgt = isinstance(weights, BlockWgt)
    if not is_blockwgt:
        missing = [p for p in pids if p not in weights.columns]
        if missing:
            raise ValueError(f"weights DataFrame missing columns for: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        weights = weights.reindex(snp_info.index).fillna(0.0)[pids]

    cis_numer = np.zeros(len(pids))
    trans_numer = np.zeros(len(pids))
    cis_denom = np.zeros(len(pids))
    trans_denom = np.zeros(len(pids))

    block_iter = snp_info.groupby("Block")
    for rg_idx, (block_idx, block_info) in enumerate(
        tqdm(block_iter, desc="Computing PWAS Z", disable=not verbose)
    ):
        block_snps = block_info.index
        if is_blockwgt:
            block_w = weights.read_block(rg_idx, columns=pids)
        else:
            block_w = weights.loc[block_snps].values

        freq = block_info["A1Freq"].values
        scale = np.sqrt(2 * freq * (1 - freq))
        block_w = block_w * scale[:, None]

        block_z = gwas_z.loc[block_snps].values

        snp_chrom = block_info["Chrom"].values
        snp_pos = block_info["PhysPos"].values
        chrom_match = snp_chrom[:, None] == prot_chrom[None, :]
        cis_mask = chrom_match & (
            snp_pos[:, None] >= (prot_start - cis_window)[None, :]
        ) & (snp_pos[:, None] <= (prot_end + cis_window)[None, :])

        w_cis = block_w * cis_mask
        w_trans = block_w * ~cis_mask

        cis_numer += w_cis.T @ block_z
        trans_numer += w_trans.T @ block_z

        eig = ldm.read_block_eig(int(block_idx))
        proj_cis = eig["U"].T @ w_cis
        proj_trans = eig["U"].T @ w_trans
        cis_denom += (proj_cis ** 2 * eig["lambdas"][:, None]).sum(axis=0)
        trans_denom += (proj_trans ** 2 * eig["lambdas"][:, None]).sum(axis=0)

    return pd.DataFrame({
        "ID": pids,
        "CIS_Z": cis_numer / np.sqrt(np.maximum(cis_denom, 1e-30)),
        "TRANS_Z": trans_numer / np.sqrt(np.maximum(trans_denom, 1e-30)),
    })


def compute_coreg(
    wgt_paths: dict[str, str],
    ldm_dir: str,
    protein_annot: pd.DataFrame,
    wgt_index_col: str = None,
    cis_window=1e6,
    subsets=("cis", "cisgenic", "cisnongenic", "cisexonic", "cisnonexonic", "trans"),
):
    """Compute co-regulation matrices (w.T @ LD @ w) for each SNP subset.

    Parameters
    ----------
    wgt_paths : dict[str, str]
        {protein_id: path_to_weights} mapping.
    ldm_dir : str
        Path to LD matrix directory.
    protein_annot : pd.DataFrame
        Protein annotation with columns CHROM, START, END, EXONS.
    wgt_index_col : str, optional
        Column name in weight files to use as index.
    cis_window : float
        Cis window in bp.
    subsets : tuple of str
        SNP subsets to compute separately.

    Returns
    -------
    dict[str, pd.DataFrame]
        {subset_name: coregulation_matrix} mapping.
    """
    ldm = BlockLDM(ldm_dir)
    snp_info = pd.read_csv(f"{ldm_dir}/snp.info", sep="\t", index_col="ID")
    wgt_reader = MultiTableReader(wgt_paths, value_col="BETA", index_col=wgt_index_col)

    subsets = list(subsets)
    result_dict = {mask: 0.0 for mask in subsets}

    for block_idx, block_info in tqdm(snp_info.groupby("Block")):
        nsnp = len(block_info)
        wgt_mat = wgt_reader.next(nsnp)
        if wgt_index_col is None:
            wgt_mat.index = block_info.index

        assert np.array_equal(wgt_mat.index, block_info.index)
        freq = block_info["A1Freq"].values
        wgt_mat *= np.sqrt(2 * freq * (1 - freq))[:, None]

        mask_dict = _build_cis_trans_masks(
            block_info, protein_annot, wgt_mat.columns, cis_window, subsets
        )

        for mask in subsets:
            result_dict[mask] += ldm.block_cov(idx=block_idx, mat=wgt_mat * mask_dict[mask])

    return {
        mask: pd.DataFrame(result_dict[mask], index=wgt_mat.columns, columns=wgt_mat.columns)
        for mask in subsets
    }


def compute_var(
    wgt_paths: dict[str, str],
    ldm_dir: str,
    protein_annot: pd.DataFrame,
    wgt_index_col: str = None,
    cis_window=1e6,
):
    """Compute variance explained (w.T @ LD @ w diagonal) for cis and trans components.

    Parameters
    ----------
    wgt_paths : dict[str, str]
        {protein_id: path_to_weights} mapping.
    ldm_dir : str
        Path to LD matrix directory.
    protein_annot : pd.DataFrame
        Protein annotation with columns CHROM, START, END.
    wgt_index_col : str, optional
        Column name in weight files to use as index.
    cis_window : float
        Cis window in bp.

    Returns
    -------
    cis_var, trans_var : pd.Series
        Variance explained by cis and trans components per protein.
    """
    ldm = BlockLDM(ldm_dir)
    snp_info = pd.read_csv(f"{ldm_dir}/snp.info", sep="\t", index_col="ID")
    wgt_reader = MultiTableReader(wgt_paths, value_col="BETA", index_col=wgt_index_col)

    result_dict = {"cis": 0.0, "trans": 0.0}

    for block_idx, block_info in tqdm(snp_info.groupby("Block")):
        nsnp = len(block_info)
        wgt_mat = wgt_reader.next(nsnp)
        if wgt_index_col is None:
            wgt_mat.index = block_info.index

        assert np.array_equal(wgt_mat.index, block_info.index)
        freq = block_info["A1Freq"].values
        wgt_mat *= np.sqrt(2 * freq * (1 - freq))[:, None]

        cis_mask = np.zeros_like(wgt_mat)
        for i, pid in enumerate(wgt_mat.columns):
            chrom, start, stop = protein_annot.loc[pid, ["CHROM", "START", "END"]]
            cis_mask[:, i] = (block_info["Chrom"] == chrom) & (
                block_info["PhysPos"].between(start - cis_window, stop + cis_window)
            )

        for mask, mask_mat in [("cis", cis_mask), ("trans", 1 - cis_mask)]:
            result_dict[mask] += ldm.block_qf(idx=block_idx, mat=wgt_mat * mask_mat)

    cis_var = pd.Series(result_dict["cis"], index=wgt_mat.columns)
    trans_var = pd.Series(result_dict["trans"], index=wgt_mat.columns)
    return cis_var, trans_var
