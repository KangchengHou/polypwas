"""Extract genotype data for protein prediction and accuracy evaluation.

Genotype and individual lists are reused from TRANS-PWAS and symlinked into
analyses/external/:
    external/genotype/  -> TRANS-PWAS/protein-prediction/DATA/genotype/
    external/*.indivlist -> TRANS-PWAS/protein-prediction/DATA/*.indivlist

This script can regenerate them from scratch if needed.
"""

import submitit
import pandas as pd
import numpy as np
import os
import subprocess
import tempfile
from pathlib import Path

EXTERNAL = Path(__file__).parent.parent / "external"
PROTEIN_PFILE = "/n/groups/price/UKBiobank/UKBPPP/DATA/genotype/imputed/merged"
UNRELATED_PATH = "/n/groups/price/martin/LDSPEC_data/UKBimp_337K_MAF001/unrelated_337K.txt"
SNPLIST = "/n/groups/price/UKBiobank/UKBPPP/DATA/ukbEUR_Imputed/snplist.txt"
BGEN_DIR = "/n/groups/price/UKBiobank/download_500K"

GENOTYPE_DIR = EXTERNAL / "ukb" / "genotype_impute+acc"
GENOTYPE_DIR = GENOTYPE_DIR


def extract_indiv():
    """Create individual lists for accuracy evaluation and variance-explained analysis."""
    unrelated_indiv = list(
        pd.read_csv(UNRELATED_PATH, sep=r"\s+", header=None)[0].values
    )
    protein_indiv = list(
        pd.read_csv(f"{PROTEIN_PFILE}.psam", sep="\t", index_col=0).index.values
    )
    print(f"{len(protein_indiv)} individuals in UKB-PPP")

    # Accuracy set: UKB-PPP individuals NOT in unrelated white British
    acc_indiv = list(set(protein_indiv) - set(unrelated_indiv))
    print(f"{len(acc_indiv)} individuals in UKB-PPP and not in white British unrelated")
    with open(GENOTYPE_DIR / "acc.indivlist", "w") as f:
        f.write("\n".join([f"{i}\t{i}" for i in acc_indiv]))

    # Imputation set: unrelated white British NOT in UKB-PPP (for variance explained)
    trait_df = pd.read_csv(
        EXTERNAL / "gwas" / "trait_values.tsv", sep="\t", index_col=0,
    )
    trait_df = trait_df.loc[
        (~trait_df.index.isin(protein_indiv)) & (trait_df.index.isin(unrelated_indiv))
    ]
    print(f"{len(trait_df)} individuals not in UKB-PPP and in white British unrelated")
    np.random.seed(42)
    impute_indiv = np.random.choice(trait_df.index, 50000, replace=False)
    with open(GENOTYPE_DIR / "impute.indivlist", "w") as f:
        f.write("\n".join([f"{i}\t{i}" for i in impute_indiv]))

    # Combined set
    assert len(acc_indiv) + len(impute_indiv) == len(set(acc_indiv) | set(impute_indiv))
    with open(GENOTYPE_DIR / "acc+impute.indivlist", "w") as f:
        f.write("\n".join([f"{i}\t{i}" for i in list(acc_indiv) + list(impute_indiv)]) + "\n")


def extract_genotype(chrom: int):
    """Extract genotype for one chromosome from BGEN to PGEN format."""
    geno_dir = str(GENOTYPE_DIR)
    os.makedirs(geno_dir, exist_ok=True)
    cmds = [
        "plink2",
        f"--bgen {BGEN_DIR}/ukb_imp_chr{chrom}_v3.bgen ref-unknown",
        f"--sample {BGEN_DIR}/ukb14048_imp_chr1_v3_s487395.sample",
        f"--extract {SNPLIST}",
        f"--keep {GENOTYPE_DIR / 'acc+impute.indivlist'}",
        "--rm-dup force-first",
        "--maj-ref",
        "--threads 4",
        "--memory 60000",
        "--make-pgen erase-dosage",
        f"--out {geno_dir}/chr{chrom}",
    ]
    subprocess.run(" ".join(cmds), shell=True)


def merge_chromosomes():
    """Merge per-chromosome PGEN files into one."""
    geno_dir = str(GENOTYPE_DIR)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
        for i in range(1, 23):
            f.write(f"chr{i}\n")
        f.flush()
        subprocess.run(
            " ".join([
                "plink2",
                f"--pmerge-list {f.name}",
                f"--pmerge-list-dir {geno_dir}",
                "--freq",
                f"--make-pgen --out {geno_dir}/merged",
            ]),
            shell=True,
        )


if __name__ == "__main__":
    if not (GENOTYPE_DIR / "acc+impute.indivlist").exists():
        extract_indiv()

    unfinished_chroms = [
        i for i in range(1, 23)
        if not (GENOTYPE_DIR / f"chr{i}.pgen").exists()
    ]
    print(f"Unfinished chroms: {unfinished_chroms}")

    if len(unfinished_chroms) > 0:
        executor = submitit.SlurmExecutor(folder="./submitit-logs")
        executor.update_parameters(
            time=900, mem="64G", partition="medium",
            cpus_per_task=4, srun_args=["--export=ALL"],
            account="price",
        )
        jobs = executor.map_array(extract_genotype, unfinished_chroms)
        print(f"Submitted {len(jobs)} jobs: {jobs[0].job_id}")
    else:
        if not (GENOTYPE_DIR / "merged.pgen").exists():
            merge_chromosomes()
        else:
            print("All done: merged genotype exists")
