"""Compute PPC-corrected PWAS Z-scores.

For each n_ppc, regresses top predicted protein PCs out of the co-regulation
matrix and PWAS covariances, then computes Z = cov / sqrt(diag(coreg)).
"""

import time
import submitit
import numpy as np
import pandas as pd
from pathlib import Path

from polypwas.utils import normalize_coreg
from polypwas.stats import compute_coreg_pc, regress_coreg_pc

EXTERNAL = Path(__file__).parent.parent / "external"
DATA_DIR = Path(__file__).parent / "DATA"


def run_ppc_pwas(group: str, gwas_group: str = "price2"):
    t0 = time.time()
    dataset, ldm, annot = group.split(".")

    # Load GWAS sample sizes
    gwas_dir = EXTERNAL / "gwas" / f"{gwas_group}_compiled"
    trait_n = {}
    for ma_path in sorted(gwas_dir.glob("*.ma")):
        trait_n[ma_path.stem] = pd.read_csv(ma_path, sep="\t", nrows=100)["N"].mean()

    # Restrict to independent traits
    indep_traits = pd.read_csv(
        EXTERNAL / "gwas" / "indep_gwas_traits.tsv", sep="\t", index_col=0,
    ).index
    trait_n = {t: trait_n[t] for t in trait_n if t in indep_traits}
    print(f"{len(trait_n)} independent traits")

    # Load PWAS covariances
    pwas_df = pd.read_csv(
        DATA_DIR / "pwas" / gwas_group / f"{group}.tsv.gz", sep="\t",
    )

    # Load and normalize coreg matrices, compute PCs
    max_pc = 30
    coreg = {}
    coreg_diag = {}
    eigvals = {}
    eigvecs = {}
    all_subsets = ["cis", "trans", "cis_nohotspot", "trans_nohotspot", "cis_noblood", "trans_noblood",
                   "cisproximal", "cisnonproximal"]
    for subset in all_subsets:
        coreg_path = DATA_DIR / "coreg" / f"{group}.{subset}.parquet"
        if not coreg_path.exists():
            print(f"  Skipping {subset}: coreg not found")
            continue
        C = pd.read_parquet(coreg_path)
        valid = np.diag(C) > 0
        C = C.loc[valid, valid]
        coreg_diag[subset] = np.diag(C).copy()
        C = normalize_coreg(C)
        coreg[subset] = C
        vals, vecs = compute_coreg_pc(C.values, n_pc=max_pc)
        eigvals[subset] = vals
        eigvecs[subset] = vecs
    available_subsets = [s for s in all_subsets if s in coreg]
    # Use pairs: (cis, trans) and (cis_nohotspot, trans_nohotspot)
    subset_pairs = [("cis", "trans")]
    if "cis_nohotspot" in coreg:
        subset_pairs.append(("cis_nohotspot", "trans_nohotspot"))
    if "cis_noblood" in coreg:
        subset_pairs.append(("cis_noblood", "trans_noblood"))
    if "cisproximal" in coreg:
        subset_pairs.append(("cisproximal", "cisnonproximal"))
    print(f"Loaded coreg for {available_subsets}, {time.time() - t0:.0f}s")

    # Compute Z-scores for each n_ppc
    results = []
    for n_ppc in [0, 10, 20, 30]:
        t1 = time.time()

        for trait in trait_n:
            if trait not in pwas_df["TRAIT"].values:
                continue

            trait_pwas = pwas_df[pwas_df["TRAIT"] == trait].set_index("ID")

            for cis_key, trans_key in subset_pairs:
                z = {}
                for subset, cov_col in [(cis_key, f"{cis_key.upper()}_COV"),
                                         (trans_key, f"{trans_key.upper()}_COV")]:
                    pids = coreg[subset].index
                    cov = trait_pwas[cov_col].reindex(pids).fillna(0).values

                    # Convert COV to Z using unnormalized coreg diagonal
                    raw_z = cov / np.sqrt(np.maximum(coreg_diag[subset], 1e-12))

                    # Regress PCs from Z-scores
                    z[subset] = pd.Series(
                        regress_coreg_pc(
                            eigvals=eigvals[subset][:n_ppc],
                            eigvecs=eigvecs[subset][:, :n_ppc],
                            z=raw_z,
                        ),
                        index=pids,
                    )

                # Align to the larger index of the pair
                all_pids = coreg[trans_key].index
                col_map = {
                    ("cis", "trans"): ("CIS_Z", "TRANS_Z"),
                    ("cis_nohotspot", "trans_nohotspot"): ("CIS_Z_NOHOTSPOT", "TRANS_Z_NOHOTSPOT"),
                    ("cis_noblood", "trans_noblood"): ("CIS_Z_NOBLOOD", "TRANS_Z_NOBLOOD"),
                    ("cisproximal", "cisnonproximal"): ("CISPROXIMAL_Z", "CISNONPROXIMAL_Z"),
                }
                col1, col2 = col_map[(cis_key, trans_key)]
                results.append(pd.DataFrame({
                    "TRAIT": trait,
                    "ID": all_pids,
                    "N_PPC": n_ppc,
                    col1: z[cis_key].reindex(all_pids),
                    col2: z[trans_key].reindex(all_pids),
                }))

        print(f"  n_ppc={n_ppc}: {time.time() - t1:.0f}s")

    # Merge results: each trait×n_ppc may have multiple DataFrames (one per subset pair)
    # Group by shared columns (TRAIT, ID, N_PPC) and merge
    from functools import reduce
    result_dfs = []
    # Group results by (trait, n_ppc) — each group has len(subset_pairs) entries
    chunk_size = len(subset_pairs)
    for i in range(0, len(results), chunk_size):
        chunk = results[i:i+chunk_size]
        merged = reduce(lambda a, b: pd.merge(a, b, on=["TRAIT", "ID", "N_PPC"]), chunk)
        result_dfs.append(merged)
    result_df = pd.concat(result_dfs, ignore_index=True)
    out_dir = DATA_DIR / "ppc_pwas"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(
        out_dir / f"{group}.tsv", sep="\t", index=False, float_format="%.6g",
    )
    print(f"Done: {group} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    groups = [
        "ukbsun.imputed.baseline+cis+pqtl",
        "ukb_linreg_0pc.imputed.baseline+cis+pqtl",
        "ukb_linreg_20pc.imputed.baseline+cis+pqtl",
        "decode.imputed.baseline+cis+pqtl",
        "csf.imputed.baseline+cis+pqtl",
    ]

    executor = submitit.SlurmExecutor(folder="./submitit-logs")
    executor.update_parameters(
        time=60, mem="16G", partition="short",
        cpus_per_task=1, srun_args=["--export=ALL"],
        account="price",
    )
    jobs = []
    with executor.batch():
        for group in groups:
            out_path = DATA_DIR / "ppc_pwas" / f"{group}.tsv"
            if out_path.exists():
                continue
            # Skip if coreg not ready
            if not (DATA_DIR / "coreg" / f"{group}.cis.parquet").exists():
                print(f"  [skipped] {group}: coreg not ready")
                continue
            jobs.append((group, executor.submit(run_ppc_pwas, group)))
    for name, job in jobs:
        print(f"  [submitted] {name}: {job.job_id}")
