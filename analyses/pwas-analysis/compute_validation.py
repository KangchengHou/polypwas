"""Validate PWAS results against burden test and PoPS.

Computes enrichment (log relative risk) of burden-significant and PoPS-top
genes among PWAS hits at different Z-score thresholds and PPC levels.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from polypwas.stats import binom_ratio

EXTERNAL = Path(__file__).parent.parent / "external"
DATA_DIR = Path(__file__).parent / "DATA"


def compile_validation_data(dataset: str):
    """Compile burden + PoPS validation data for a dataset."""
    burden_df = (
        pd.read_csv(EXTERNAL / "gwas" / "burden.tsv", sep="\t")
        .assign(
            BURDEN_PVALUE=lambda x: 2 * stats.norm.sf(np.abs(x["beta"] / x["standard_error"])),
            BURDEN_ZSCORE=lambda x: x["beta"] / x["standard_error"],
        )
        .rename(columns={"trait_id": "TRAIT", "ensg": "ENSEMBL"})
    )[["TRAIT", "ENSEMBL", "BURDEN_ZSCORE", "BURDEN_PVALUE"]]

    pops_df = (
        pd.read_csv(EXTERNAL / "gwas" / "pops.tsv", sep="\t")
        .rename(columns={"trait_id": "TRAIT", "ENSGID": "ENSEMBL", "PoPS_Score": "POPS_SCORE"})
    )[["TRAIT", "ENSEMBL", "POPS_SCORE"]]

    gene_df = pd.read_csv(
        EXTERNAL / "pqtl" / "sumstats" / f"{dataset}.gene.tsv", sep="\t",
    )[["ID", "ENSEMBL"]]

    gene_df = pd.merge(gene_df, burden_df, on="ENSEMBL")
    gene_df = pd.merge(gene_df, pops_df, on=["TRAIT", "ENSEMBL"])

    # Define hits per trait
    gene_df["BURDEN_SIGNIF"] = 0
    gene_df["POPS_TOP"] = 0
    for trait, trait_df in gene_df.groupby("TRAIT"):
        burden_signif = trait_df.BURDEN_PVALUE < 0.05 / len(trait_df)
        gene_df.loc[trait_df.index[burden_signif], "BURDEN_SIGNIF"] = 1
        gene_df.loc[
            trait_df.nlargest(burden_signif.sum(), "POPS_SCORE").index, "POPS_TOP"
        ] = 1

    return gene_df[["TRAIT", "ID", "ENSEMBL", "BURDEN_ZSCORE", "BURDEN_PVALUE",
                     "POPS_SCORE", "BURDEN_SIGNIF", "POPS_TOP"]]


def compute_validation_stats(group: str):
    """Compute validation enrichment stats for one group."""
    dataset, ldm, annot = group.split(".")

    pwas_df = pd.read_csv(DATA_DIR / "ppc_pwas" / f"{group}.tsv", sep="\t")
    valid_df = compile_validation_data(dataset)
    print(f"{group}: {len(pwas_df)} PWAS rows, {len(valid_df)} validation rows")

    merged = pd.merge(valid_df, pwas_df, on=["TRAIT", "ID"])
    print(f"  Merged: {len(merged)} rows")

    merged["ABS_CIS_LEVEL"] = pd.cut(
        merged.CIS_Z.abs(), bins=[-1, 5, 10, np.inf],
        labels=["<5", "5-10", ">=10"],
    )
    merged["ABS_TRANS_LEVEL"] = pd.cut(
        merged.TRANS_Z.abs(), bins=[-1, 5, 10, np.inf],
        labels=["<5", "5-10", ">=10"],
    )

    binary_stats = []
    multi_stats = []
    multi_cis_stats = []

    for n_ppc, ppc_df in merged.groupby("N_PPC"):
        # Binary: |cis|>10 & |trans|>10
        subset = ppc_df[(ppc_df.CIS_Z.abs() > 10) & (ppc_df.TRANS_Z.abs() > 10)]
        for target in ["POPS_TOP", "BURDEN_SIGNIF"]:
            logrr, logrr_se = binom_ratio(
                x1=int(subset[target].sum()), n1=len(subset),
                x2=int(ppc_df[target].sum()), n2=len(ppc_df),
            )
            binary_stats.append({
                "group": group, "n_ppc": n_ppc, "target_var": target,
                "n_hit": int(subset[target].sum()), "n_subset": len(subset),
                "logrr": logrr, "logrr_se": logrr_se,
            })

        # Multi: by cis × trans level
        for (cis_level, trans_level), subset in ppc_df.groupby(
            ["ABS_CIS_LEVEL", "ABS_TRANS_LEVEL"], observed=False,
        ):
            for target in ["POPS_TOP", "BURDEN_SIGNIF"]:
                logrr, logrr_se = binom_ratio(
                    x1=int(subset[target].sum()), n1=len(subset),
                    x2=int(ppc_df[target].sum()), n2=len(ppc_df),
                )
                multi_stats.append({
                    "group": group, "n_ppc": n_ppc, "target_var": target,
                    "cis_level": cis_level, "trans_level": trans_level,
                    "n_hit": int(subset[target].sum()), "n_subset": len(subset),
                    "logrr": logrr, "logrr_se": logrr_se,
                })

        # Multi cis only
        for cis_level, subset in ppc_df.groupby("ABS_CIS_LEVEL", observed=False):
            for target in ["POPS_TOP", "BURDEN_SIGNIF"]:
                logrr, logrr_se = binom_ratio(
                    x1=int(subset[target].sum()), n1=len(subset),
                    x2=int(ppc_df[target].sum()), n2=len(ppc_df),
                )
                multi_cis_stats.append({
                    "group": group, "n_ppc": n_ppc, "target_var": target,
                    "cis_level": cis_level,
                    "n_hit": int(subset[target].sum()), "n_subset": len(subset),
                    "logrr": logrr, "logrr_se": logrr_se,
                })

    out_dir = DATA_DIR / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(binary_stats).to_csv(
        out_dir / f"binary.{group}.tsv", sep="\t", index=False, float_format="%.6g",
    )
    pd.DataFrame(multi_stats).to_csv(
        out_dir / f"multi.{group}.tsv", sep="\t", index=False, float_format="%.6g",
    )
    pd.DataFrame(multi_cis_stats).to_csv(
        out_dir / f"multi_cis.{group}.tsv", sep="\t", index=False, float_format="%.6g",
    )
    print(f"  Done: {group}")


if __name__ == "__main__":
    groups = [
        "ukbsun.imputed.baseline+cis+pqtl",
        "ukb_linreg_0pc.imputed.baseline+cis+pqtl",
        "ukb_linreg_20pc.imputed.baseline+cis+pqtl",
    ]
    for group in groups:
        compute_validation_stats(group)
