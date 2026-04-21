"""Evaluate prediction accuracy: correlate predicted vs observed protein levels.

Computes per-protein correlation between predicted and observed protein levels
for held-out accuracy individuals. Predictions are optionally residualized
against their top predicted protein PCs (PPCs) before computing correlations.

One job per base group (e.g. ukb_linreg_0pc.imputed.baseline+cis+pqtl),
iterating over all subsets and n_pc values.
"""

import time
import submitit
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.linalg import eigh
from tqdm import tqdm
from pathlib import Path

from polypwas.utils import inverse_rank_normalize

EXTERNAL = Path(__file__).parent.parent / "external"
PRED_DIR = Path(__file__).parent / "DATA" / "prediction"
UKB_DATA_DIR = "/n/groups/price/UKBiobank/UKBPPP/DATA/pqtl"
SUBSETS = ["cis", "cisgenic", "cisnongenic", "cisproximal", "cisnonproximal", "cisexonic", "cisnonexonic", "trans",
           "cis_nohotspot", "trans_nohotspot", "cis_noblood", "trans_noblood"]


def load_ukb_protein():
    """Load residualized UKB protein phenotypes for accuracy evaluation."""
    protein_df = pd.read_csv(
        f"{UKB_DATA_DIR}/protein.pheno", sep="\t", index_col=0,
    ).drop(columns=["IID"])
    covar_df = pd.read_csv(
        f"{UKB_DATA_DIR}/protein.covar", sep="\t", index_col=0,
    ).drop(columns=["IID"])

    with open(EXTERNAL / "ukb" / "acc.indivlist", "r") as f:
        valid_indiv = [int(line.split("\t")[0]) for line in f.read().splitlines()]

    # Restrict to white British
    wb_indiv = (
        pd.read_csv(
            "/n/groups/price/UKBiobank/UKBPPP/DATA/trait-value.tsv",
            sep="\t", index_col=0,
        )
        .query("cov_ETHNICITY == 1001")
        .index.values
    )
    valid_indiv = list(set(valid_indiv) & set(wb_indiv))
    print(f"{len(valid_indiv)} individuals in white British accuracy set")

    covar_cols = [col for col in covar_df.columns if not col.startswith("PROTPC")]
    pheno_df = pd.merge(
        protein_df.loc[valid_indiv, :],
        covar_df.loc[valid_indiv, :],
        left_index=True, right_index=True,
    )

    resid_df = {}
    for pid in tqdm(protein_df.columns, desc="Residualizing proteins"):
        normalized = inverse_rank_normalize(pheno_df[pid])
        resid = sm.OLS(
            normalized, sm.add_constant(pheno_df[covar_cols]), missing="drop",
        ).fit().resid
        resid_df[pid] = resid
    return pd.DataFrame(resid_df)


def series_cor(s1, s2):
    """Correlation between two series, handling NaN."""
    idx = s1.dropna().index.intersection(s2.dropna().index)
    return np.corrcoef(s1.loc[idx], s2.loc[idx])[0, 1]


def eval_accuracy(group: str):
    """Compute prediction accuracy for one base group across all subsets and n_pc values."""
    t0 = time.time()
    ukb_resid_df = load_ukb_protein()
    acc_indiv = ukb_resid_df.index
    print(f"Loaded observed proteins in {time.time() - t0:.0f}s")

    dataset, ldm, annot = group.split(".")
    gene_info_df = pd.read_csv(
        EXTERNAL / "pqtl" / "sumstats" / f"{dataset}.gene.tsv",
        sep="\t", index_col=0,
    )

    stats_df = []
    for subset in SUBSETS:
        pred_path = PRED_DIR / f"{group}.{subset}.parquet"
        if not pred_path.exists():
            print(f"  Skipping {subset}: {pred_path} not found")
            continue

        score_df = pd.read_parquet(pred_path)
        score_df = score_df.loc[:, score_df.std() > 1e-12]
        print(f"  {subset}: {score_df.shape}, {time.time() - t0:.0f}s")

        # PCA on all individuals (not just acc) to match PAPER behavior
        score_std = (score_df - score_df.mean(axis=0)) / score_df.std(axis=0)
        Xnp = score_std.values
        n_prot = Xnp.shape[1]
        max_pc = 30
        _, eigvecs = eigh(Xnp.T @ Xnp, subset_by_index=[n_prot - max_pc, n_prot - 1])

        for n_pc in [0, 10, 20, 30]:
            t1 = time.time()
            if n_pc == 0:
                resid_score_df = score_df.loc[score_df.index.isin(acc_indiv), :]
            else:
                pcs = pd.DataFrame(Xnp @ eigvecs[:, -n_pc:], index=score_df.index)
                resid_score_df = sm.OLS(score_df, sm.add_constant(pcs)).fit().resid
                resid_score_df = resid_score_df.loc[resid_score_df.index.isin(acc_indiv), :]

            for pid in resid_score_df.columns:
                uniprot = gene_info_df.loc[pid, "UNIPROT"]
                if uniprot not in ukb_resid_df.columns:
                    continue
                corr = series_cor(resid_score_df[pid], ukb_resid_df[uniprot])
                stats_df.append({
                    "group": f"{group}.{subset}", "n_pc": n_pc,
                    "dataset": dataset, "ldm": ldm, "annot": annot,
                    "subset": subset, "pid": pid, "uniprot": uniprot,
                    "corr": corr,
                })
            print(f"    n_pc={n_pc}: {time.time() - t1:.0f}s")

    stats_df = pd.DataFrame(stats_df)
    out_dir = Path("DATA/accuracy")
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(
        out_dir / f"{group}.tsv", sep="\t", index=False, float_format="%.6g",
    )
    print(f"Done: {group} in {time.time() - t0:.0f}s")
    print(stats_df.groupby(["subset", "n_pc"])["corr"].apply(lambda x: f"median R²={( x**2).median():.4f}"))


if __name__ == "__main__":
    groups = [
        "ukb_linreg_0pc.imputed.baseline+cis+pqtl",
        "ukb_linreg_0pc.hm3.baseline+cis+pqtl",
    ]

    executor = submitit.SlurmExecutor(folder="./submitit-logs")
    executor.update_parameters(
        time=60, mem="40G", partition="short",
        cpus_per_task=1, srun_args=["--export=ALL"],
        account="price",
    )
    jobs = []
    with executor.batch():
        for group in groups:
            out_path = Path(f"DATA/accuracy/{group}.tsv")
            if out_path.exists():
                continue
            jobs.append((group, executor.submit(eval_accuracy, group)))
    for name, job in jobs:
        print(f"  [submitted] {name}: {job.job_id}")
