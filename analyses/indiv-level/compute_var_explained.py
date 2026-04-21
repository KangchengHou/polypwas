"""Compute variance explained by predicted proteins for GWAS traits.

One SLURM job per trait. Uses eigh for PCA (60x faster than SVD) and
cho_solve for R² (16x faster than pinvh). Matches TRANS-PWAS PAPER logic:
drop NaN individuals before PCA.
"""

import time
import submitit
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import eigh
from pathlib import Path

from polypwas.stats import adjusted_rsquared

EXTERNAL = Path(__file__).parent.parent / "external"
PRED_DIR = Path(__file__).parent / "DATA" / "prediction"
MASKS = ["cis", "cisgenic", "cisnongenic", "cisproximal", "cisnonproximal", "cisexonic", "cisnonexonic", "trans",
         "cis_nohotspot", "trans_nohotspot", "cis_noblood", "trans_noblood"]
COMBINED = [("cis_trans_both", "cis", "trans"),
            ("cis_genic_both", "cisgenic", "cisnongenic"),
            ("cis_proximal_both", "cisproximal", "cisnonproximal"),
            ("cis_exonic_both", "cisexonic", "cisnonexonic"),
            ("cis_trans_nohotspot_both", "cis_nohotspot", "trans_nohotspot"),
            ("cis_trans_noblood_both", "cis_noblood", "trans_noblood")]


def fast_pca_resid(X: pd.DataFrame, n_pc: int, eigvecs_all: np.ndarray = None) -> pd.DataFrame:
    """Residualize X against its top PCs using eigh on X.T @ X.

    Uses eigh (60x faster than full SVD) to get PCs, then OLS to residualize
    (includes intercept, matching sklearn PCA + OLS behavior).

    If eigvecs_all is provided, slices the last n_pc columns instead of
    recomputing the eigendecomposition.
    """
    Xnp = X.values
    if eigvecs_all is None:
        n_prot = Xnp.shape[1]
        _, eigvecs = eigh(Xnp.T @ Xnp, subset_by_index=[n_prot - n_pc, n_prot - 1])
    else:
        eigvecs = eigvecs_all[:, -n_pc:]
    pcs = pd.DataFrame(Xnp @ eigvecs, index=X.index)
    return sm.OLS(X, sm.add_constant(pcs)).fit().resid


def run_var_explained(group: str, trait: str):
    """Compute variance explained for one group × trait."""
    t0 = time.time()

    # Load individual list
    with open(EXTERNAL / "ukb" / "impute.indivlist", "r") as f:
        impute_indiv = [int(line.split("\t")[0]) for line in f.readlines()]

    # Load trait values
    trait_df = pd.read_csv(
        EXTERNAL / "gwas" / "trait_values.tsv", sep="\t", index_col=0,
    )
    trait_df = trait_df.loc[trait_df.index.isin(impute_indiv), :]
    covar_cols = ["cov_SEX", "cov_AGE", "cov_AGE_SQ"]
    data_df = pd.concat([trait_df[trait], trait_df[covar_cols]], axis=1).dropna()
    covar_cols = [col for col in covar_cols if data_df[col].var() > 0]

    # Regress out covariates and standardize
    data_df[trait] = sm.OLS(data_df[trait], sm.add_constant(data_df[covar_cols])).fit().resid
    data_df[trait] = (data_df[trait] - data_df[trait].mean()) / data_df[trait].std()

    # Load predictions for each mask, drop NaN before standardizing (matches PAPER logic)
    pred_df_dict = {}
    for mask in MASKS:
        pred_path = PRED_DIR / f"{group}.{mask}.parquet"
        pred = pd.read_parquet(pred_path).loc[data_df.index, :]
        pred = pred.loc[:, pred.std() > 1e-12]
        pred_df_dict[mask] = (pred - pred.mean(axis=0)) / pred.std(axis=0)

    print(f"Loaded data: {len(data_df)} indiv, {pred_df_dict['cis'].shape[1]} proteins, {time.time() - t0:.1f}s")

    # Precompute eigenvectors (top 30) once per mask to avoid redundant eigh calls
    max_pc = 30
    eigvecs_dict = {}
    for mask in MASKS:
        Xnp = pred_df_dict[mask].values
        n_prot = Xnp.shape[1]
        _, eigvecs_dict[mask] = eigh(Xnp.T @ Xnp, subset_by_index=[n_prot - max_pc, n_prot - 1])

    # Compute R² for each n_pc
    stats_df = {}
    for n_pc in [0, 10, 20, 30]:
        t1 = time.time()
        if n_pc == 0:
            resid_pred = {mask: pred_df_dict[mask].copy() for mask in MASKS}
        else:
            resid_pred = {}
            for mask in MASKS:
                resid_pred[mask] = fast_pca_resid(pred_df_dict[mask], n_pc, eigvecs_dict[mask])

        # Rename columns to avoid collisions when concatenating
        for mask in resid_pred:
            resid_pred[mask].columns = [f"{p}_{mask}" for p in resid_pred[mask].columns]

        y = data_df[trait].values
        stats = {}
        for mask in MASKS:
            est, se = adjusted_rsquared(y, sm.add_constant(resid_pred[mask]).values)
            stats[f"{mask}_est"] = est
            stats[f"{mask}_se"] = se

        for name, m1, m2 in COMBINED:
            combined = pd.concat([resid_pred[m1], resid_pred[m2]], axis=1)
            est, se = adjusted_rsquared(y, sm.add_constant(combined).values)
            stats[f"{name}_est"] = est
            stats[f"{name}_se"] = se

        stats_df[n_pc] = stats
        print(f"  n_pc={n_pc}: {time.time() - t1:.1f}s")

    stats_df = pd.DataFrame(stats_df).T
    stats_df.index.name = "n_ppc"
    stats_df = stats_df.reset_index()
    out_dir = Path("DATA/var_explained")
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(
        out_dir / f"{group}.{trait}.tsv", sep="\t", index=False, float_format="%.6g",
    )
    print(f"Done: {group}.{trait} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    groups = [
        "ukb_linreg_0pc.imputed.baseline+cis+pqtl",
        "ukb_linreg_0pc.hm3.baseline+cis+pqtl",
    ]
    all_traits = pd.read_csv(
        EXTERNAL / "gwas" / "indep_gwas_traits.tsv", sep="\t", index_col=0,
    ).index.tolist()
    trait_cols = pd.read_csv(
        EXTERNAL / "gwas" / "trait_values.tsv", sep="\t", index_col=0, nrows=0,
    ).columns.tolist()
    traits = [t for t in all_traits if t in trait_cols]

    executor = submitit.SlurmExecutor(folder="./submitit-logs")
    executor.update_parameters(
        time=60, mem="40G", partition="short",
        cpus_per_task=1, srun_args=["--export=ALL"],
        account="price",
    )

    jobs = []
    with executor.batch():
        for group in groups:
            for trait in traits:
                out_path = Path(f"DATA/var_explained/{group}.{trait}.tsv")
                if out_path.exists():
                    continue
                jobs.append((f"{group}.{trait}", executor.submit(run_var_explained, group, trait)))
    print(f"Submitted {len(jobs)} jobs")
    for name, job in jobs[:5]:
        print(f"  {name}: {job.job_id}")
    if len(jobs) > 5:
        print(f"  ... and {len(jobs) - 5} more")
