"""Convert SBayesRC weights and GWAS sumstats to BlockWgt parquet format."""

import submitit
import os
from glob import glob
from pathlib import Path

from polypwas.ld import BlockLDM
from polypwas.store import BlockWgt

EXTERNAL = Path(__file__).parent.parent / "external"
LDM_DIRS = {
    "hm3": str(EXTERNAL / "ldm" / "ukbEUR_HM3"),
    "imputed": str(EXTERNAL / "ldm" / "ukbEUR_Imputed"),
}
OUT_DIR = EXTERNAL / "blockwgt"


def convert_sbayesrc(group: str):
    """Convert one SBayesRC weight group to parquet."""
    out_path = str(OUT_DIR / "sbayesrc" / f"{group}.parquet")
    if os.path.exists(out_path):
        print(f"Skipping {group}: already exists")
        return

    ldm_name = group.split(".")[1]  # e.g. "imputed" or "hm3"
    ldm = BlockLDM(LDM_DIRS[ldm_name])
    wgt_paths = {
        os.path.basename(p).replace(".tsv.gz", ""): p
        for p in sorted(glob(str(EXTERNAL / "pqtl" / "sbayesrc" / group / "*.tsv.gz")))
    }
    print(f"Converting {group}: {len(wgt_paths)} proteins, {ldm.n_snp:,} SNPs")
    BlockWgt.from_weight_files(wgt_paths, ldm.snp_info, out_path)
    print(f"Done: {out_path} ({os.path.getsize(out_path) / 1e9:.1f} GB)")


def convert_gwas(gwas_group: str, ldm_name: str):
    """Convert one GWAS group to parquet."""
    out_path = str(OUT_DIR / "gwas" / f"{gwas_group}.parquet")
    if os.path.exists(out_path):
        print(f"Skipping {gwas_group}: already exists")
        return

    ldm = BlockLDM(LDM_DIRS[ldm_name])
    gwas_paths = {
        os.path.basename(p).replace(".ma", ""): p
        for p in sorted(glob(str(EXTERNAL / "gwas" / f"{gwas_group}_compiled" / "*.ma")))
    }
    print(f"Converting {gwas_group}: {len(gwas_paths)} traits, {ldm.n_snp:,} SNPs")
    BlockWgt.from_gwas_files(gwas_paths, ldm.snp_info, out_path)
    print(f"Done: {out_path} ({os.path.getsize(out_path) / 1e9:.1f} GB)")


if __name__ == "__main__":
    sbayesrc_groups = [
        # imputed (7.3M SNPs)
        "ukbsun.imputed.baseline+cis+pqtl",
        "ukbsun.imputed.none",
        "ukb_linreg_0pc.imputed.baseline+cis+pqtl",
        "ukb_linreg_0pc.imputed.baseline+cis",
        "ukb_linreg_0pc.imputed.none",
        "ukb_linreg_20pc.imputed.baseline+cis+pqtl",
        # ukb_linreg_20pc.imputed.baseline+cis: no SBayesRC weights trained
        "ukb_linreg_20pc.imputed.none",
        "decode.imputed.baseline+cis+pqtl",
        "csf.imputed.baseline+cis+pqtl",
        # hm3 (1.15M SNPs)
        "ukbsun.hm3.baseline+cis+pqtl",
        "ukbsun.hm3.none",
        "ukb_linreg_0pc.hm3.baseline+cis+pqtl",
        "ukb_linreg_0pc.hm3.baseline+cis",
        "ukb_linreg_0pc.hm3.none",
        "ukb_linreg_20pc.hm3.baseline+cis+pqtl",
        "ukb_linreg_20pc.hm3.baseline+cis",
        "ukb_linreg_20pc.hm3.none",
        "ukb_linreg_100pc.hm3.baseline+cis+pqtl",
        "decode.hm3.baseline+cis+pqtl",
        "csf.hm3.baseline+cis+pqtl",
    ]
    gwas_groups = [
        ("price2", "imputed"),  # 88 traits, 7.3M SNPs
        ("pass", "hm3"),        # 394 traits, 1.15M SNPs
    ]

    executor = submitit.SlurmExecutor(folder="./submitit-logs")
    executor.update_parameters(
        time=3000, mem="64G", partition="medium",
        cpus_per_task=1, srun_args=["--export=ALL"],
        account="price",
    )

    jobs = []
    with executor.batch():
        for group in sbayesrc_groups:
            out_path = OUT_DIR / "sbayesrc" / f"{group}.parquet"
            if out_path.exists():
                print(f"Skipping {group}: already exists")
                continue
            jobs.append((f"sbayesrc/{group}", executor.submit(convert_sbayesrc, group)))

        for gwas_group, ldm_name in gwas_groups:
            out_path = OUT_DIR / "gwas" / f"{gwas_group}.parquet"
            if out_path.exists():
                print(f"Skipping {gwas_group}: already exists")
                continue
            jobs.append((f"gwas/{gwas_group}", executor.submit(convert_gwas, gwas_group, ldm_name)))

    for name, job in jobs:
        print(f"  [submitted] {name}: {job.job_id}")
