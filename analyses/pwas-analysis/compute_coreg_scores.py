"""Compute co-regulation scores from co-regulation matrices."""

import numpy as np
import pandas as pd
from pathlib import Path

from polypwas.utils import normalize_coreg
from polypwas.stats import compute_coreg_pc, regress_coreg_pc

EXTERNAL = Path(__file__).parent.parent / "external"


def run_coreg_scores(group: str):
    coreg_stats_df = []
    for subset in ["cis", "trans"]:
        coreg = pd.read_parquet(f"DATA/coreg/{group}.{subset}.parquet")
        valid = np.diag(coreg) > 0
        coreg = normalize_coreg(coreg.loc[valid, valid])
        eigvals, eigvecs = compute_coreg_pc(coreg.values, n_pc=30)

        for n_pc in [0, 10, 20, 30]:
            if n_pc == 0:
                coreg_score = (coreg**2).sum(axis=1)
            else:
                residual = regress_coreg_pc(
                    eigvals=eigvals[:n_pc], eigvecs=eigvecs[:, :n_pc],
                    coreg=coreg.values,
                )
                coreg_score = pd.Series(
                    (residual**2).sum(axis=1), index=coreg.index,
                )

            coreg_stats_df.append(pd.DataFrame({
                "group": group,
                "pid": coreg_score.index,
                "subset": subset,
                "n_pc": n_pc,
                "coreg_score": coreg_score.values,
            }))

    coreg_stats_df = pd.concat(coreg_stats_df, ignore_index=True)
    out_dir = Path("DATA/coreg_scores")
    out_dir.mkdir(parents=True, exist_ok=True)
    coreg_stats_df.to_csv(
        out_dir / f"{group}.tsv", sep="\t", index=False,
        na_rep="NA", float_format="%.6g",
    )


if __name__ == "__main__":
    groups = [
        "ukbsun.imputed.baseline+cis+pqtl",
        # "decode.imputed.baseline+cis+pqtl",
        # "csf.imputed.baseline+cis+pqtl",
    ]
    for group in groups:
        run_coreg_scores(group)
