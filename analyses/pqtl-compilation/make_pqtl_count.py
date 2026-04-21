"""Create per-SNP pQTL count annotation for hotspot filtering.

For each SNP, counts how many proteins have a genome-wide significant
(p < 5e-8) cis or trans pQTL. Output columns: cis_count, trans_count,
total_count. SNPs with total_count > 50 are considered pQTL hotspots.
"""

import submitit
import pandas as pd
from glob import glob
from pathlib import Path

from polypwas.sbayesrc import summarize_signif_pqtl

EXTERNAL = Path(__file__).parent.parent / "external"


def make_pqtl_count(dataset: str):
    gene_info = pd.read_csv(
        EXTERNAL / "pqtl" / "sumstats" / f"{dataset}.gene.tsv",
        sep="\t", index_col=0,
    )
    snp_info = pd.read_csv(
        EXTERNAL / "ldm" / "ukbEUR_Imputed" / "snp.info",
        sep="\t", index_col="ID",
    )
    ma_list = sorted(glob(
        str(EXTERNAL / "pqtl" / "sumstats" / dataset / "*.ma.gz")
    ))
    print(f"{dataset}: {len(ma_list)} proteins, {len(snp_info)} SNPs")

    count_df = summarize_signif_pqtl(
        ma_list=ma_list,
        gene_info=gene_info,
        snp_info=snp_info,
        verbose=True,
    )
    count_df = count_df.rename(columns={"cis": "cis_count", "trans": "trans_count"})
    count_df["total_count"] = count_df["cis_count"] + count_df["trans_count"]

    out_path = EXTERNAL / "pqtl" / "sumstats" / f"{dataset}.pqtl_count.tsv"
    count_df.to_csv(out_path, sep="\t")
    print(f"Wrote {out_path}: {(count_df['total_count'] > 0).sum()} SNPs with any pQTL")
    print(f"  Hotspots (total > 50): {(count_df['total_count'] > 50).sum()} SNPs")


if __name__ == "__main__":
    datasets = ["ukbsun", "ukb_linreg_0pc", "ukb_linreg_20pc", "decode", "csf"]

    executor = submitit.SlurmExecutor(folder="./submitit-logs")
    executor.update_parameters(
        time=600, mem="32G", partition="short",
        cpus_per_task=1, srun_args=["--export=ALL"],
        account="price",
    )
    jobs = []
    with executor.batch():
        for dataset in datasets:
            out_path = EXTERNAL / "pqtl" / "sumstats" / f"{dataset}.pqtl_count.tsv"
            if out_path.exists():
                continue
            jobs.append((dataset, executor.submit(make_pqtl_count, dataset)))
    for name, job in jobs:
        print(f"  [submitted] {name}: {job.job_id}")
