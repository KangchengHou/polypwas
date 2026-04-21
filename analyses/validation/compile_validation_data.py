"""Compile validation data: merge burden test + PoPS for each dataset."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

EXTERNAL = Path(__file__).parent.parent / "external"


def compile_validation_data():
    burden_df = (
        pd.read_csv(EXTERNAL / "gwas" / "burden.tsv", sep="\t")
        .assign(
            BURDEN_PVALUE=lambda x: 2 * stats.norm.sf(np.abs(x["beta"] / x["standard_error"])),
            BURDEN_ZSCORE=lambda x: x["beta"] / x["standard_error"],
        )
        .rename(columns={"trait_id": "TRAIT", "ensg": "ENSEMBL"})
    )[["TRAIT", "ENSEMBL", "BURDEN_ZSCORE", "BURDEN_PVALUE"]]

    pops_df = pd.read_csv(EXTERNAL / "gwas" / "pops.tsv", sep="\t").rename(
        columns={"trait_id": "TRAIT", "ENSGID": "ENSEMBL", "PoPS_Score": "POPS_SCORE"},
    )[["TRAIT", "ENSEMBL", "POPS_SCORE"]]

    out_dir = Path("DATA/validation_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in ["ukbsun", "decode", "csf", "ukb_linreg_0pc", "ukb_linreg_20pc"]:
        gene_df = pd.read_csv(
            EXTERNAL / "pqtl" / "sumstats" / f"{dataset}.gene.tsv", sep="\t",
        )[["ID", "ASSAY", "ENSEMBL"]]
        print(f"#proteins for {dataset}: {len(gene_df.ID.unique())}")

        gene_df = pd.merge(gene_df, burden_df, on="ENSEMBL")
        print(f"  after burden: {len(gene_df.ID.unique())}")

        gene_df = pd.merge(gene_df, pops_df, on=["TRAIT", "ENSEMBL"])
        print(f"  after pops: {len(gene_df.ID.unique())}")

        gene_df["BURDEN_SIGNIF"] = 0
        gene_df["POPS_TOP"] = 0
        for trait, trait_df in gene_df.groupby("TRAIT"):
            burden_signif = trait_df.BURDEN_PVALUE < 0.05 / len(trait_df)
            gene_df.loc[trait_df.index[burden_signif], "BURDEN_SIGNIF"] = 1
            gene_df.loc[
                trait_df.nlargest(burden_signif.sum(), "POPS_SCORE").index, "POPS_TOP"
            ] = 1

        gene_df = gene_df[[
            "TRAIT", "ID", "ASSAY", "ENSEMBL",
            "BURDEN_ZSCORE", "BURDEN_PVALUE", "POPS_SCORE",
            "BURDEN_SIGNIF", "POPS_TOP",
        ]]
        gene_df.to_csv(
            out_dir / f"{dataset}.tsv", sep="\t", index=False, float_format="%.6g",
        )


if __name__ == "__main__":
    compile_validation_data()
