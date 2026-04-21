"""SBayesRC training, summary statistics munging, and pQTL annotation."""

import numpy as np
import pandas as pd
import scipy.stats
import subprocess
import tempfile
import shutil
import os
from .config import get_config


def _thread_env(threads: int | None) -> dict[str, str] | None:
    """Build subprocess environment for optional OpenMP thread control."""
    if threads is None:
        return None
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    return env


def validate_runtime(rscript_path: str | None = None) -> str:
    """Validate that Rscript exists and SBayesRC is installed."""
    rscript_path = get_config("Rscript") if rscript_path is None else rscript_path
    expr = (
        "if(requireNamespace('SBayesRC', quietly=TRUE)) cat('SBAYESRC_OK\\n') else quit(status=1)"
    )
    try:
        result = subprocess.run(
            [rscript_path, "-e", expr],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Rscript not found: {rscript_path}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or "SBayesRC package check failed"
        raise RuntimeError(
            f"SBayesRC runtime validation failed for {rscript_path}: {detail}"
        ) from exc

    if "SBAYESRC_OK" not in result.stdout:
        raise RuntimeError(f"SBayesRC runtime validation failed for {rscript_path}")
    return rscript_path


def train(ma_path, ldm_dir, annot_path, out_prefix, threads: int | None = None):
    """Train SBayesRC model.

    Parameters
    ----------
    ma_path : str
        Path to formatted summary statistics (.ma format).
    ldm_dir : str
        Path to LD matrix directory.
    annot_path : str or None
        Optional path to annotation file.
    out_prefix : str
        Output prefix for trained weights.
    threads : int or None
        Optional thread count passed via OMP_NUM_THREADS.
    """
    rscript_path = validate_runtime()
    args = [
        f"mafile='{ma_path}'",
        f"LDdir='{ldm_dir}'",
        f"outPrefix='{out_prefix}'",
        "bTune=FALSE",
        "log2file=FALSE",
    ]
    if annot_path is not None:
        args.insert(3, f"annot='{annot_path}'")
    expr = f"SBayesRC::sbayesrc({', '.join(args)})"
    subprocess.run([rscript_path, "-e", expr], check=True, env=_thread_env(threads))


def munge_sumstats(
    path: str, out: str, ldm_dir: str, input_format: str, threads: int | None = None
):
    """Format, tidy, and impute summary statistics for SBayesRC.

    Supports 'plink2' and 'ldsc' input formats. Outputs imputed .ma file.

    Parameters
    ----------
    path : str
        Path to input summary statistics.
    out : str
        Path for output imputed summary statistics.
    ldm_dir : str
        Path to LD matrix directory (used for allele matching and imputation).
    input_format : str
        One of 'plink2' or 'ldsc'.
    threads : int or None
        Optional thread count passed via OMP_NUM_THREADS.
    """
    rscript_path = get_config("Rscript")

    if input_format == "plink2":
        column_dict = {
            "ID": "SNP",
            "ALT": "A1",
            "REF": "A2",
            "A1_FREQ": "freq",
            "BETA": "b",
            "SE": "se",
            "P": "p",
            "OBS_CT": "N",
        }
        gwas = pd.read_csv(
            path,
            sep="\t",
            usecols=list(column_dict.keys()),
        ).rename(columns=column_dict)
    elif input_format == "ldsc":
        gwas = pd.read_csv(path, sep="\t", index_col="SNP")
        ldm_snp_info = pd.read_csv(f"{ldm_dir}/snp.info", sep="\t", index_col="ID")
        gwas = gwas[gwas.index.isin(ldm_snp_info.index)]

        match_idx = (gwas["A1"] == ldm_snp_info.loc[gwas.index, "A1"]) & (
            gwas["A2"] == ldm_snp_info.loc[gwas.index, "A2"]
        )
        flip_idx = (gwas["A1"] == ldm_snp_info.loc[gwas.index, "A2"]) & (
            gwas["A2"] == ldm_snp_info.loc[gwas.index, "A1"]
        )

        gwas.loc[match_idx, "freq"] = ldm_snp_info.loc[gwas.index[match_idx], "A1Freq"]
        gwas.loc[flip_idx, "freq"] = 1 - ldm_snp_info.loc[gwas.index[flip_idx], "A1Freq"]
        gwas = gwas[match_idx | flip_idx]

        gwas["se"] = 1 / np.sqrt(gwas["N"] * 2 * gwas["freq"] * (1 - gwas["freq"]))
        gwas["b"] = gwas["Z"] * gwas["se"]
        gwas["p"] = scipy.stats.norm.sf(np.abs(gwas["Z"])) * 2
        gwas = gwas[["A1", "A2", "freq", "b", "se", "p", "N"]].reset_index()
    else:
        raise ValueError(f"Input format {input_format} not supported")

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_prefix = f"{tmp_dir}/gwas"
        gwas.to_csv(out_prefix + ".raw_ma", sep="\t", index=False)
        subprocess.run(
            (
                f'{rscript_path} -e "SBayesRC::tidy('
                f"mafile='{out_prefix}.raw_ma', "
                f"LDdir='{ldm_dir}', "
                f"N_sd_range=6, "
                f"output='{out_prefix}.tidy_ma', "
                f'log2file=FALSE)"'
            ),
            shell=True,
            env=_thread_env(threads),
        )
        subprocess.run(
            (
                f'{rscript_path} -e "SBayesRC::impute('
                f"mafile='{out_prefix}.tidy_ma', "
                f"LDdir='{ldm_dir}', "
                f"output='{out_prefix}.imp_ma', "
                f'log2file=FALSE)"'
            ),
            shell=True,
            env=_thread_env(threads),
        )
        shutil.move(out_prefix + ".imp_ma", out)


def summarize_signif_pqtl(
    ma_list: list[str],
    gene_info: pd.DataFrame,
    snp_info: pd.DataFrame,
    cis_window: float = 1e6,
    signif_thresh: float = 5e-8,
    verbose: bool = False,
):
    """Make pQTL annotation (cis/trans counts per SNP) for SBayesRC.

    Parameters
    ----------
    ma_list : list[str]
        Paths to per-protein summary statistics (.ma.gz).
    gene_info : pd.DataFrame
        Gene locations indexed by protein ID, with columns CHROM, START, END.
    snp_info : pd.DataFrame
        SNP info indexed by SNP ID, with columns Chrom, PhysPos.
    cis_window : float
        Cis window in bp (default 1e6).
    signif_thresh : float
        P-value threshold for significant pQTLs (default 5e-8).
    verbose : bool
        Print per-protein progress.

    Returns
    -------
    pd.DataFrame
        Columns ['cis', 'trans'] with counts of significant pQTLs per SNP.
    """
    signif_annot = pd.DataFrame(0, index=snp_info.index.rename("SNP"), columns=["cis", "trans"])

    for i, path in enumerate(ma_list):
        pid = os.path.basename(path).split(".ma.gz")[0]
        chrom, start, stop = gene_info.loc[pid, ["CHROM", "START", "END"]]
        ma_df = pd.read_csv(path, sep="\t", usecols=["b", "se"])
        pvalues = pd.Series(
            scipy.stats.norm.sf(np.abs(ma_df["b"] / ma_df["se"])) * 2,
            index=snp_info.index,
        )
        signif_snps = pvalues[pvalues < signif_thresh].index.values
        cis_mask = (snp_info.loc[signif_snps, "Chrom"] == chrom) & (
            snp_info.loc[signif_snps, "PhysPos"].between(start - cis_window, stop + cis_window)
        )
        cis_snps, trans_snps = signif_snps[cis_mask], signif_snps[~cis_mask]
        signif_annot.loc[cis_snps, "cis"] += 1
        signif_annot.loc[trans_snps, "trans"] += 1

        if verbose:
            print(
                f"[{i + 1}/{len(ma_list)}] {pid}: "
                f"{len(cis_snps)} cis, {len(trans_snps)} trans (P < {signif_thresh})"
            )
    return signif_annot
