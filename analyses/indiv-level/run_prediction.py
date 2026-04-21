"""Compute predicted protein levels using pgenlib + BlockWgt.

Reads genotype block-by-block via pgenlib, multiplies by SBayesRC weights
from BlockWgt parquet, applying cis/trans masks. All proteins scored in a
single pass (~60 min for 7.3M SNPs × 63K individuals × 2.8K proteins).
"""

import time
import submitit
import pgenlib
import numpy as np
import pandas as pd
from pathlib import Path

from polypwas.store import BlockWgt
from polypwas.pwas import _build_cis_trans_masks, load_hotspot_snps, load_exclude_snps

EXTERNAL = Path(__file__).parent.parent / "external"
GENOTYPE_DIR = EXTERNAL / "ukb" / "genotype_impute+acc"
SNP_SUBSETS = ["cis", "cisgenic", "cisnongenic", "cisproximal", "cisnonproximal", "cisexonic", "cisnonexonic", "trans",
               "cis_nohotspot", "trans_nohotspot", "cis_noblood", "trans_noblood"]


def run_prediction(group: str, subsets: list[str] = SNP_SUBSETS, cis_window: float = 1e6):
    """Score all proteins for a group using pgenlib + BlockWgt."""
    dataset, ldm_name, annot = group.split(".")
    ldm_dir = EXTERNAL / "ldm" / {"imputed": "ukbEUR_Imputed", "hm3": "ukbEUR_HM3"}[ldm_name]

    # Load metadata
    pvar = pd.read_csv(GENOTYPE_DIR / "merged.pvar", sep="\t")
    psam = pd.read_csv(GENOTYPE_DIR / "merged.psam", sep="\t")
    n_snp, n_indiv = len(pvar), len(psam)
    pvar_id_to_idx = pd.Series(np.arange(n_snp), index=pvar["ID"].values)

    snp_info = pd.read_csv(ldm_dir / "snp.info", sep="\t", index_col="ID")
    wgt_store = BlockWgt(str(EXTERNAL / "blockwgt" / "sbayesrc" / f"{group}.parquet"))
    pids = wgt_store.columns
    protein_annot = pd.read_csv(
        EXTERNAL / "pqtl" / "sumstats" / f"{dataset}.gene.tsv", sep="\t", index_col=0,
    )

    # Load exclusion SNP sets
    exclude_snps = {
        "nohotspot": load_hotspot_snps(
            EXTERNAL / "pqtl" / "sumstats" / f"{dataset}.pqtl_count.tsv",
        ),
        "noblood": load_exclude_snps(EXTERNAL / "misc" / "blood_cell_gws_snps.txt"),
    }
    for k, v in exclude_snps.items():
        print(f"Loaded {len(v)} {k} SNPs")

    # Precompute allele flip: pgenlib counts ALT, weights are for snp_info A1
    pvar_alt = pvar.set_index("ID")["ALT"]
    flip = (pvar_alt.loc[snp_info.index].values != snp_info["A1"].values).astype(np.float32)
    flip = np.where(flip, -1.0, 1.0)  # -1 for flip, +1 for no flip

    # Initialize accumulators
    result = {s: np.zeros((n_indiv, len(pids)), dtype=np.float32) for s in subsets}

    # Open PGEN reader
    reader = pgenlib.PgenReader(str(GENOTYPE_DIR / "merged.pgen").encode(), raw_sample_ct=n_indiv)

    block_groups = list(snp_info.groupby("Block"))
    flip_offset = 0
    t_total = time.time()
    for i, (block_idx, block_info) in enumerate(block_groups):
        t0 = time.time()
        pgen_indices = pvar_id_to_idx.loc[block_info.index].values
        n_block_snp = len(pgen_indices)

        # Read genotype: (n_snp, n_indiv), impute missing (-9) with per-SNP mean
        buf = np.empty((n_block_snp, n_indiv), dtype=np.int8)
        for j, idx in enumerate(pgen_indices):
            reader.read(int(idx), buf[j])
        G = buf.astype(np.float32)
        miss_mask = G == -9
        if miss_mask.any():
            G[miss_mask] = 0.0
            miss_count = miss_mask.sum(axis=1, keepdims=True)
            row_mean = G.sum(axis=1, keepdims=True) / np.maximum(n_indiv - miss_count, 1)
            G += miss_mask * row_mean
        G = G.T  # (n_indiv, n_snp)

        # Read weights; apply allele flip to W (smaller than G)
        W = wgt_store.read_block(i)
        block_flip = flip[flip_offset:flip_offset + n_block_snp]
        flip_offset += n_block_snp
        W = W * block_flip[:, None]

        # Build masks
        mask_dict = _build_cis_trans_masks(block_info, protein_annot, pids, cis_window, subsets,
                                          exclude_snps=exclude_snps)

        # Accumulate predictions
        for s in subsets:
            result[s] += G @ (W * mask_dict[s])

        elapsed = time.time() - t0
        print(f"Block {i}/{len(block_groups)} ({n_block_snp} SNPs) {elapsed:.1f}s")

    reader.close()
    print(f"Total prediction time: {time.time() - t_total:.0f}s")

    # Write output
    out_dir = Path("DATA/prediction")
    out_dir.mkdir(parents=True, exist_ok=True)
    iids = psam["IID"].values
    for s in subsets:
        df = pd.DataFrame(result[s], index=iids, columns=pids)
        df.index.name = "IID"
        df.to_parquet(out_dir / f"{group}.{s}.parquet", index=True)
        print(f"Wrote {out_dir / f'{group}.{s}.parquet'}")


if __name__ == "__main__":
    groups = [
        "ukb_linreg_0pc.imputed.baseline+cis+pqtl",
        "ukb_linreg_0pc.hm3.baseline+cis+pqtl",
        # "ukbsun.imputed.baseline+cis+pqtl",
        # "decode.imputed.baseline+cis+pqtl",
        # "csf.imputed.baseline+cis+pqtl",
    ]
    # One job per (group, subset) for parallelism
    job_args = [(g, [s]) for g in groups for s in SNP_SUBSETS]

    executor = submitit.SlurmExecutor(folder="./submitit-logs")
    executor.update_parameters(
        time=180, mem="48G", partition="short",
        cpus_per_task=8, srun_args=["--export=ALL"],
        account="price",
    )
    jobs = executor.map_array(
        run_prediction,
        [a[0] for a in job_args],
        [a[1] for a in job_args],
    )
    print(f"Submitted {len(jobs)} jobs: {jobs[0].job_id}")
    for (g, s), job in zip(job_args, jobs):
        print(f"  {g}.{s[0]}: {job.job_id}")
