"""Compute PWAS covariances (numerator of Z-scores) for all proteins × traits."""

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


def run_pwas(group: str, gwas_group: str):
    dataset, ldm_name, annot = group.split(".")
    wgt_store = BlockWgt(str(EXTERNAL / "blockwgt" / "sbayesrc" / f"{group}.parquet"))
    gwas_store = BlockWgt(str(EXTERNAL / "blockwgt" / "gwas" / f"{gwas_group}.parquet"))
    ldm = BlockLDM(LDM_DIRS[ldm_name])
    snp_info = ldm.snp_info
    protein_annot = pd.read_csv(
        EXTERNAL / "pqtl" / "sumstats" / f"{dataset}.gene.tsv", sep="\t", index_col=0,
    )

    # For hm3: GWAS blockwgt is built for imputed panel, need to subset per block
    need_gwas_subset = (ldm_name == "hm3")
    if need_gwas_subset:
        imp_snp_info = pd.read_csv(
            str(EXTERNAL / "ldm" / "ukbEUR_Imputed" / "snp.info"), sep="\t", index_col="ID",
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
    traits = gwas_store.columns
    result_dict = {(mask, "numer"): 0.0 for mask in SUBSETS}

    imp_block_groups = list(imp_snp_info.groupby("Block")) if need_gwas_subset else None
    gwas_block_i = 0
    for i, (block_idx, block_info) in enumerate(snp_info.groupby("Block")):
        W = wgt_store.read_block(i)

        if need_gwas_subset:
            # Read full imputed block, subset to hm3 SNPs
            Z_full = gwas_store.read_block(gwas_block_i)
            gwas_block_i += 1
            imp_block_snps = imp_block_groups[i][1].index
            hm3_idx = np.isin(imp_block_snps, block_info.index)
            Z = Z_full[hm3_idx]
        else:
            Z = gwas_store.read_block(i)

        freq = block_info["A1Freq"].values
        W *= np.sqrt(2 * freq * (1 - freq))[:, None]

        mask_dict = _build_cis_trans_masks(
            block_info, protein_annot, pids, 1e6, SUBSETS,
            exclude_snps=exclude_snps,
        )

        for mask in SUBSETS:
            result_dict[(mask, "numer")] += (W * mask_dict[mask]).T @ Z

    # Reshape to long format
    cov_dfs = {}
    for mask in SUBSETS:
        mat = result_dict[(mask, "numer")]
        df = pd.DataFrame(mat, index=pids, columns=traits)
        df = df.reset_index(names="ID").melt(
            id_vars="ID", var_name="TRAIT", value_name=f"{mask.upper()}_COV",
        )
        cov_dfs[mask] = df

    long_df = cov_dfs[SUBSETS[0]]
    for mask in SUBSETS[1:]:
        long_df = pd.merge(long_df, cov_dfs[mask], on=["TRAIT", "ID"])
    long_df = long_df[["TRAIT", "ID", *[f"{mask.upper()}_COV" for mask in SUBSETS]]]

    out_dir = Path("DATA/pwas") / gwas_group
    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_dir / f"{group}.tsv.gz", sep="\t", index=False, float_format="%.6g")


if __name__ == "__main__":
    groups = [
        "ukbsun.imputed.baseline+cis+pqtl",
        "ukb_linreg_0pc.imputed.baseline+cis+pqtl",
        "ukb_linreg_20pc.imputed.baseline+cis+pqtl",
        "decode.imputed.baseline+cis+pqtl",
        "csf.imputed.baseline+cis+pqtl",
        "ukb_linreg_0pc.hm3.baseline+cis+pqtl",
    ]
    gwas_groups = ["price2"]

    executor = submitit.SlurmExecutor(folder="./submitit-logs")
    executor.update_parameters(
        time=2880, mem="64G", partition="medium",
        cpus_per_task=1, srun_args=["--export=ALL"],
        account="price",
    )
    jobs = []
    with executor.batch():
        for gwas_group in gwas_groups:
            for group in groups:
                out_path = Path(f"DATA/pwas/{gwas_group}/{group}.tsv.gz")
                if out_path.exists():
                    continue
                jobs.append((f"{gwas_group}/{group}", executor.submit(run_pwas, group, gwas_group)))
    for name, job in jobs:
        print(f"  [submitted] {name}: {job.job_id}")
