"""Smoke test: load real data, compute PWAS z-score for one protein × one trait."""

import numpy as np
import pandas as pd
from pathlib import Path

EXTERNAL = Path(__file__).parent.parent / "analyses" / "external"
LDM_DIR = EXTERNAL / "ldm" / "ukbEUR_Imputed"
PQTL_GROUP = "ukbsun.imputed.baseline+cis+pqtl"
SBAYESRC_DIR = EXTERNAL / "pqtl" / "sbayesrc" / PQTL_GROUP
GWAS_DIR = EXTERNAL / "gwas" / "price2_compiled"
GENE_TSV = EXTERNAL / "pqtl" / "sumstats" / "ukbsun.gene.tsv"


def test_blockldm_loads():
    """Test that BlockLDM can read snp.info and ldm.info."""
    from polypwas.ld import BlockLDM

    ldm = BlockLDM(str(LDM_DIR))
    assert ldm.n_snp > 1_000_000, f"Expected >1M SNPs, got {ldm.n_snp}"
    assert ldm.n_block > 0
    print(f"BlockLDM: {ldm.n_snp:,} SNPs, {ldm.n_block} blocks")


def test_read_block_eig():
    """Test reading a single block eigendecomposition."""
    from polypwas.ld import BlockLDM

    ldm = BlockLDM(str(LDM_DIR))
    first_block = ldm.ldm_info.index[0]
    eig = ldm.read_block_eig(first_block)
    assert eig["m"] > 0
    assert eig["k"] > 0
    assert eig["U"].shape == (eig["m"], eig["k"])
    assert eig["lambdas"].shape == (eig["k"],)
    print(f"Block {first_block}: {eig['m']} SNPs, {eig['k']} eigenvectors")


def test_single_pwas_zscore():
    """End-to-end: compute PWAS z-score for one protein × one trait."""
    from polypwas.ld import BlockLDM

    ldm = BlockLDM(str(LDM_DIR))
    snp_info = ldm.snp_info

    # Pick first protein with weights
    wgt_files = sorted(SBAYESRC_DIR.glob("*.tsv.gz"))
    pid = wgt_files[0].name.replace(".tsv.gz", "")  # e.g. A1BG:P04217:OID30771:v1
    print(f"Protein: {pid}")

    # Read weights
    wgt = pd.read_csv(wgt_files[0], sep="\t")["BETA"].values
    assert len(wgt) == ldm.n_snp, f"Weight length {len(wgt)} != {ldm.n_snp} SNPs"
    wgt = pd.Series(wgt, index=snp_info.index)

    # Read gene annotation for cis masking
    gene_df = pd.read_csv(GENE_TSV, sep="\t", index_col="ID")
    chrom, start, end = gene_df.loc[pid, ["CHROM", "START", "END"]]

    # Read one GWAS trait
    trait = "biochemistry_LDLdirect"
    gwas = pd.read_csv(GWAS_DIR / f"{trait}.ma", sep="\t", index_col="SNP")
    gwasz = (gwas.loc[snp_info.index, "b"] / gwas.loc[snp_info.index, "se"]).astype(float)

    # Scale weights by sqrt(2pq)
    freq = snp_info["A1Freq"].values
    wgt_scaled = wgt * np.sqrt(2 * freq * (1 - freq))

    # Cis mask
    cis_mask = (snp_info["Chrom"] == chrom) & snp_info["PhysPos"].between(start - 1e6, end + 1e6)
    trans_mask = ~cis_mask

    # Compute block-wise: numerator (w'z) and denominator (w'Rw)
    cis_numer = 0.0
    trans_numer = 0.0
    cis_denom = 0.0
    trans_denom = 0.0

    for block_idx in ldm.ldm_info.index[:20]:  # first 20 blocks for speed
        block_snps = snp_info[snp_info["Block"] == block_idx].index
        bw_cis = (wgt_scaled.loc[block_snps] * cis_mask.loc[block_snps]).values
        bw_trans = (wgt_scaled.loc[block_snps] * trans_mask.loc[block_snps]).values
        bz = gwasz.loc[block_snps].values

        cis_numer += bw_cis @ bz
        trans_numer += bw_trans @ bz

        block_ldm_mat = ldm.read_block_mat(block_idx)
        cis_denom += bw_cis @ block_ldm_mat @ bw_cis
        trans_denom += bw_trans @ block_ldm_mat @ bw_trans

    cis_z = cis_numer / np.sqrt(max(cis_denom, 1e-30))
    trans_z = trans_numer / np.sqrt(max(trans_denom, 1e-30))

    print(f"Trait: {trait}")
    print(f"CIS_Z  = {cis_z:.4f} (numer={cis_numer:.4g}, denom={cis_denom:.4g})")
    print(f"TRANS_Z = {trans_z:.4f} (numer={trans_numer:.4g}, denom={trans_denom:.4g})")

    # Sanity checks
    assert np.isfinite(cis_z), "CIS_Z is not finite"
    assert np.isfinite(trans_z), "TRANS_Z is not finite"
    assert cis_denom >= 0, "CIS denominator should be non-negative"
    assert trans_denom >= 0, "TRANS denominator should be non-negative"
    print("PASS")


if __name__ == "__main__":
    test_blockldm_loads()
    test_read_block_eig()
    test_single_pwas_zscore()
