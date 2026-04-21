"""Compute validation statistics: enrichment of PWAS hits for burden/PoPS."""

import numpy as np
import pandas as pd
from pathlib import Path

from polypwas.stats import binom_ratio

PWAS_DIR = Path(__file__).parent.parent / "pwas-analysis" / "DATA"


def compute_validation_stats():
    out_dir = Path("DATA/validation_stats")
    out_dir.mkdir(parents=True, exist_ok=True)

    for group in [
        "ukbsun.imputed.baseline+cis+pqtl",
        "decode.imputed.baseline+cis+pqtl",
        "ukb_linreg_0pc.imputed.baseline+cis+pqtl",
        "ukb_linreg_20pc.imputed.baseline+cis+pqtl",
    ]:
        dataset = group.split(".")[0]
        pwas_df = pd.read_csv(
            PWAS_DIR / "ppc_pwas" / "price2" / f"{group}.tsv.gz", sep="\t",
        )
        all_valid_df = pd.read_csv(f"DATA/validation_data/{dataset}.tsv", sep="\t")
        print(f"{group}: PWAS {len(pwas_df)} rows, validation {len(all_valid_df)} rows")

        all_valid_df = pd.merge(
            all_valid_df, pwas_df, on=["TRAIT", "ID", "ASSAY", "ENSEMBL"],
        )
        print(f"  merged: {len(all_valid_df)} rows")

        all_valid_df["ABS_CIS_LEVEL"] = pd.cut(
            all_valid_df.CIS_Z.abs(),
            bins=[-1, 5, 10, np.inf], precision=1,
            labels=["<5", "5-10", ">=10"],
        )
        all_valid_df["ABS_TRANS_LEVEL"] = pd.cut(
            all_valid_df.TRANS_Z.abs(),
            bins=[-1, 5, 10, np.inf], precision=1,
            labels=["<5", "5-10", ">=10"],
        )

        binary_stats_df = []
        multi_stats_df = []
        multi_cis_stats_df = []

        for n_pc, valid_df in all_valid_df.groupby("N_PC"):
            subset_df = valid_df[
                (valid_df.TRANS_Z.abs() > 10) & (valid_df.CIS_Z.abs() > 10)
            ]
            for target_var in ["POPS_TOP", "BURDEN_SIGNIF"]:
                logrr, logrr_se = binom_ratio(
                    x1=subset_df[target_var].sum(), n1=len(subset_df),
                    x2=valid_df[target_var].sum(), n2=len(valid_df),
                )
                binary_stats_df.append({
                    "group": group,
                    "n_hit": subset_df[target_var].sum(),
                    "n_subset": len(subset_df),
                    "n_pc": n_pc,
                    "target_var": target_var,
                    "logrr": logrr,
                    "logrr_se": logrr_se,
                    "prioritized": ";".join(
                        subset_df.apply(
                            lambda row: f"{row['TRAIT']},{row['ID']},{row['ASSAY']}", axis=1,
                        )
                    ),
                    "hits": ";".join(
                        subset_df[subset_df[target_var] == 1].apply(
                            lambda row: f"{row['TRAIT']},{row['ID']},{row['ASSAY']}", axis=1,
                        )
                    ),
                })

            for (cis_level, trans_level), subset_df in valid_df.groupby(
                ["ABS_CIS_LEVEL", "ABS_TRANS_LEVEL"], observed=False,
            ):
                for target_var in ["POPS_TOP", "BURDEN_SIGNIF"]:
                    logrr, logrr_se = binom_ratio(
                        x1=subset_df[target_var].sum(), n1=len(subset_df),
                        x2=valid_df[target_var].sum(), n2=len(valid_df),
                    )
                    multi_stats_df.append({
                        "group": group,
                        "n_hit": subset_df[target_var].sum(),
                        "n_subset": len(subset_df),
                        "n_pc": n_pc,
                        "target_var": target_var,
                        "cis_level": cis_level,
                        "trans_level": trans_level,
                        "logrr": logrr,
                        "logrr_se": logrr_se,
                    })

            for cis_level, subset_df in valid_df.groupby("ABS_CIS_LEVEL", observed=False):
                for target_var in ["POPS_TOP", "BURDEN_SIGNIF"]:
                    logrr, logrr_se = binom_ratio(
                        x1=subset_df[target_var].sum(), n1=len(subset_df),
                        x2=valid_df[target_var].sum(), n2=len(valid_df),
                    )
                    multi_cis_stats_df.append({
                        "group": group,
                        "n_hit": subset_df[target_var].sum(),
                        "n_subset": len(subset_df),
                        "n_pc": n_pc,
                        "target_var": target_var,
                        "cis_level": cis_level,
                        "logrr": logrr,
                        "logrr_se": logrr_se,
                    })

        pd.DataFrame(binary_stats_df).to_csv(
            out_dir / f"binary.{group}.tsv", sep="\t", index=False, float_format="%.6g",
        )
        pd.DataFrame(multi_stats_df).to_csv(
            out_dir / f"multi.{group}.tsv", sep="\t", index=False, float_format="%.6g",
        )
        pd.DataFrame(multi_cis_stats_df).to_csv(
            out_dir / f"multi_cis.{group}.tsv", sep="\t", index=False, float_format="%.6g",
        )


if __name__ == "__main__":
    compute_validation_stats()
