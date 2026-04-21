"""Compute co-regulation matrices (w'Rw) for cis and trans components."""

import time
import submitit
import numpy as np
import pandas as pd
from pathlib import Path

from polypwas.ld import BlockLDM
from polypwas.store import BlockWgt
from polypwas.pwas import _build_cis_trans_masks, load_hotspot_snps, load_exclude_snps

EXTERNAL = Path(__file__).parent.parent / "external"
LDM_DIRS = {
    "hm3": str(EXTERNAL / "ldm" / "ukbEUR_HM3"),
    "imputed": str(EXTERNAL / "ldm" / "ukbEUR_Imputed"),
}
SUBSETS = ["cis", "cisgenic", "cisnongenic", "cisproximal", "cisnonproximal",
           "cisexonic", "cisnonexonic", "trans",
           "cis_nohotspot", "trans_nohotspot", "cis_noblood", "trans_noblood"]


def run_coreg(group: str):
    dataset, ldm_name, annot = group.split(".")
    wgt_store = BlockWgt(str(EXTERNAL / "blockwgt" / "sbayesrc" / f"{group}.parquet"))
    ldm = BlockLDM(LDM_DIRS[ldm_name])
    snp_info = ldm.snp_info
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

    pids = wgt_store.columns
    result_dict = {mask: 0.0 for mask in SUBSETS}

    block_groups = list(snp_info.groupby("Block"))
    t_total = time.time()
    for i, (block_idx, block_info) in enumerate(block_groups):
        t0 = time.time()
        W = wgt_store.read_block(i)
        freq = block_info["A1Freq"].values
        W *= np.sqrt(2 * freq * (1 - freq))[:, None]

        mask_dict = _build_cis_trans_masks(
            block_info, protein_annot, pids, 1e6, SUBSETS,
            exclude_snps=exclude_snps,
        )

        for mask in SUBSETS:
            result_dict[mask] += ldm.block_cov(idx=block_idx, mat=W * mask_dict[mask])

        elapsed = time.time() - t_total
        print(f"Block {i}/{len(block_groups)} ({time.time()-t0:.1f}s, total {elapsed/60:.0f}min)", flush=True)

    print(f"Done computing coreg in {(time.time()-t_total)/60:.0f}min", flush=True)
    out_dir = Path("DATA/coreg")
    out_dir.mkdir(parents=True, exist_ok=True)
    for mask in SUBSETS:
        df = pd.DataFrame(result_dict[mask], index=pids, columns=pids)
        df.to_parquet(out_dir / f"{group}.{mask}.parquet", index=True)


if __name__ == "__main__":
    groups = [
        "ukbsun.imputed.baseline+cis+pqtl",
        "ukb_linreg_0pc.imputed.baseline+cis+pqtl",
        "ukb_linreg_20pc.imputed.baseline+cis+pqtl",
        "decode.imputed.baseline+cis+pqtl",
        "csf.imputed.baseline+cis+pqtl",
        "ukb_linreg_0pc.hm3.baseline+cis+pqtl",
    ]

    executor = submitit.SlurmExecutor(folder="./submitit-logs")
    executor.update_parameters(
        time=2880, mem="80G", partition="medium",
        cpus_per_task=1, srun_args=["--export=ALL"],
        account="price",
    )
    jobs = []
    with executor.batch():
        for group in groups:
            out_path = Path(f"DATA/coreg/{group}.cis_noblood.parquet")
            if out_path.exists():
                continue
            jobs.append((group, executor.submit(run_coreg, group)))
    for name, job in jobs:
        print(f"  [submitted] {name}: {job.job_id}")
