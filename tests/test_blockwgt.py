"""Test BlockWgt: correctness vs old MultiTableReader + speed comparison."""

import time
import os
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
import tempfile
import sys

from polypwas.ld import BlockLDM
from polypwas.utils import MultiTableReader
from polypwas.pwas import _build_cis_trans_masks
from polypwas.store import BlockWgt

sys.stdout.reconfigure(line_buffering=True)

EXTERNAL = Path(__file__).parent.parent / "analyses" / "external"
LDM_DIR = str(EXTERNAL / "ldm" / "ukbEUR_Imputed")
GROUP = "ukbsun.imputed.baseline+cis+pqtl"
N_TEST_BLOCKS = 5
N_TEST_GWAS = 3


def setup():
    ldm = BlockLDM(LDM_DIR)
    snp_info = ldm.snp_info

    wgt_paths = {
        os.path.basename(p).replace(".tsv.gz", ""): p
        for p in sorted(glob(str(EXTERNAL / "pqtl" / "sbayesrc" / GROUP / "*.tsv.gz")))
    }
    protein_annot = pd.read_csv(
        EXTERNAL / "pqtl" / "sumstats" / "ukbsun.gene.tsv", sep="\t", index_col=0,
    )
    gwas_dir = EXTERNAL / "gwas" / "price2_compiled"
    gwas_paths = {
        p.name.replace(".ma", ""): str(p)
        for p in sorted(gwas_dir.glob("*.ma"))[:N_TEST_GWAS]
    }

    # Subset snp_info to first N_TEST_BLOCKS blocks only
    blocks = list(snp_info.groupby("Block"))
    test_block_ids = [b[0] for b in blocks[:N_TEST_BLOCKS]]
    snp_info_sub = snp_info[snp_info["Block"].isin(test_block_ids)]

    return ldm, snp_info, snp_info_sub, wgt_paths, protein_annot, gwas_paths


def run_old(ldm, snp_info_sub, wgt_paths, protein_annot, gwas_paths):
    """Run old MultiTableReader-based code on N_TEST_BLOCKS blocks."""
    # Load GWAS (subset to test SNPs)
    t0 = time.time()
    gwas_df = {}
    for trait, path in gwas_paths.items():
        df = pd.read_csv(path, sep="\t", index_col="SNP")
        gwas_df[trait] = (df.loc[snp_info_sub.index, "b"] / df.loc[snp_info_sub.index, "se"])
    gwas_df = pd.DataFrame(gwas_df)
    t_gwas = time.time() - t0

    # Read weights for test blocks only
    wgt_reader = MultiTableReader(wgt_paths, value_col="BETA")
    subsets = ["cis", "trans"]

    pwas_results = {m: 0.0 for m in subsets}
    coreg_results = {m: 0.0 for m in subsets}
    t_wgt = 0
    t_compute = 0

    for block_idx, block_info in snp_info_sub.groupby("Block"):
        nsnp = len(block_info)

        t = time.time()
        wgt_mat = wgt_reader.next(nsnp)
        wgt_mat.index = block_info.index
        freq = block_info["A1Freq"].values
        wgt_mat *= np.sqrt(2 * freq * (1 - freq))[:, None]
        t_wgt += time.time() - t

        t = time.time()
        mask_dict = _build_cis_trans_masks(
            block_info, protein_annot, wgt_mat.columns, 1e6, subsets,
        )
        gwas_mat = gwas_df.loc[block_info.index, :]
        for m in subsets:
            pwas_results[m] += (wgt_mat * mask_dict[m]).T @ gwas_mat
        for m in subsets:
            coreg_results[m] += ldm.block_cov(idx=block_idx, mat=wgt_mat * mask_dict[m])
        t_compute += time.time() - t

    return {
        "pwas": pwas_results, "coreg": coreg_results,
        "t_gwas": t_gwas, "t_wgt": t_wgt, "t_compute": t_compute,
    }


def run_new(ldm, snp_info_sub, wgt_pq, gwas_pq, protein_annot):
    """Run new BlockWgt-based code on N_TEST_BLOCKS blocks."""
    t0 = time.time()
    wgt_store = BlockWgt(wgt_pq)
    gwas_store = BlockWgt(gwas_pq)
    t_open = time.time() - t0

    pids = wgt_store.columns
    subsets = ["cis", "trans"]
    pwas_results = {m: 0.0 for m in subsets}
    coreg_results = {m: 0.0 for m in subsets}
    t_read = 0
    t_compute = 0

    for i, (block_idx, block_info) in enumerate(snp_info_sub.groupby("Block")):
        t = time.time()
        W = wgt_store.read_block(i)
        Z = gwas_store.read_block(i)
        t_read += time.time() - t

        t = time.time()
        freq = block_info["A1Freq"].values
        W *= np.sqrt(2 * freq * (1 - freq))[:, None]

        mask_dict = _build_cis_trans_masks(
            block_info, protein_annot, pids, 1e6, subsets,
        )
        for m in subsets:
            pwas_results[m] += (W * mask_dict[m]).T @ Z

        wgt_df = pd.DataFrame(W, index=block_info.index, columns=pids)
        for m in subsets:
            coreg_results[m] += ldm.block_cov(idx=block_idx, mat=wgt_df * mask_dict[m])
        t_compute += time.time() - t

    return {
        "pwas": pwas_results, "coreg": coreg_results,
        "t_open": t_open, "t_read": t_read, "t_compute": t_compute,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Loading shared data...")
    ldm, snp_info_full, snp_info_sub, wgt_paths, protein_annot, gwas_paths = setup()
    test_snps = len(snp_info_sub)
    print(f"  {len(wgt_paths)} proteins, {len(gwas_paths)} traits")
    print(f"  Testing {N_TEST_BLOCKS} blocks ({test_snps:,} SNPs)\n")

    # --- Convert 5 blocks to parquet ---
    tmpdir = tempfile.mkdtemp(prefix="blockwgt_test_")
    wgt_pq = os.path.join(tmpdir, "weights.parquet")
    gwas_pq = os.path.join(tmpdir, "gwas.parquet")

    print("Converting weights (5 blocks) to parquet...")
    t0 = time.time()
    BlockWgt.from_weight_files(wgt_paths, snp_info_sub, wgt_pq)
    t_convert_wgt = time.time() - t0
    wgt_size = os.path.getsize(wgt_pq)
    print(f"  Done in {t_convert_wgt:.0f}s, size={wgt_size / 1e6:.1f} MB")

    print("Converting GWAS (5 blocks) to parquet...")
    t0 = time.time()
    BlockWgt.from_gwas_files(gwas_paths, snp_info_sub, gwas_pq)
    t_convert_gwas = time.time() - t0
    gwas_size = os.path.getsize(gwas_pq)
    print(f"  Done in {t_convert_gwas:.0f}s, size={gwas_size / 1e6:.1f} MB")

    wgt_store = BlockWgt(wgt_pq)
    gwas_store = BlockWgt(gwas_pq)
    print(f"\n  {wgt_store}")
    print(f"  {gwas_store}\n")

    # --- Run old ---
    print("Running OLD (MultiTableReader + pd.read_csv)...")
    t0 = time.time()
    old = run_old(ldm, snp_info_sub, wgt_paths, protein_annot, gwas_paths)
    t_old = time.time() - t0
    print(f"  GWAS load:    {old['t_gwas']:.1f}s")
    print(f"  Weight read:  {old['t_wgt']:.1f}s")
    print(f"  Compute:      {old['t_compute']:.1f}s")
    print(f"  TOTAL:        {t_old:.1f}s\n")

    # --- Run new ---
    print("Running NEW (BlockWgt parquet)...")
    t0 = time.time()
    new = run_new(ldm, snp_info_sub, wgt_pq, gwas_pq, protein_annot)
    t_new = time.time() - t0
    print(f"  Open files:   {new['t_open']:.1f}s")
    print(f"  Block reads:  {new['t_read']:.1f}s")
    print(f"  Compute:      {new['t_compute']:.1f}s")
    print(f"  TOTAL:        {t_new:.1f}s\n")

    # --- Correctness ---
    print("=" * 60)
    print("CORRECTNESS CHECK")
    all_pass = True
    for m in ["cis", "trans"]:
        old_p = old["pwas"][m].values if isinstance(old["pwas"][m], pd.DataFrame) else old["pwas"][m]
        new_p = new["pwas"][m].values if isinstance(new["pwas"][m], pd.DataFrame) else new["pwas"][m]
        max_diff = np.max(np.abs(old_p - new_p))
        rel_diff = max_diff / (np.max(np.abs(old_p)) + 1e-30)
        ok = rel_diff < 1e-3
        print(f"  PWAS  {m:5s}: max_rel_diff={rel_diff:.2e} [{'PASS' if ok else 'FAIL'}]")
        all_pass &= ok

        old_c = old["coreg"][m].values if isinstance(old["coreg"][m], pd.DataFrame) else old["coreg"][m]
        new_c = new["coreg"][m].values if isinstance(new["coreg"][m], pd.DataFrame) else new["coreg"][m]
        max_diff = np.max(np.abs(old_c - new_c))
        rel_diff = max_diff / (np.max(np.abs(old_c)) + 1e-30)
        ok = rel_diff < 1e-3
        print(f"  Coreg {m:5s}: max_rel_diff={rel_diff:.2e} [{'PASS' if ok else 'FAIL'}]")
        all_pass &= ok

    # --- Speed summary ---
    print(f"\n{'=' * 60}")
    print("SPEED SUMMARY (5 blocks)")
    old_io = old["t_gwas"] + old["t_wgt"]
    new_io = new["t_open"] + new["t_read"]
    print(f"  I/O:     old={old_io:.1f}s  new={new_io:.1f}s  speedup={old_io / max(new_io, 0.01):.1f}x")
    print(f"  Compute: old={old['t_compute']:.1f}s  new={new['t_compute']:.1f}s")
    print(f"  Total:   old={t_old:.1f}s  new={t_new:.1f}s  speedup={t_old / max(t_new, 0.01):.1f}x")

    ratio = len(snp_info_full) / test_snps
    print(f"\n  Extrapolated full run ({ldm.n_block} blocks):")
    print(f"    OLD total: ~{t_old * ratio / 60:.1f} min")
    print(f"    NEW total: ~{t_new * ratio / 60:.1f} min")

    # --- Ad-hoc queries ---
    print(f"\n{'=' * 60}")
    print("AD-HOC QUERIES")
    pid = wgt_store.columns[0]
    t0 = time.time()
    col = wgt_store.read_column(pid)
    print(f"  read_column: {len(col):,} SNPs in {time.time()-t0:.3f}s")

    t0 = time.time()
    region = wgt_store.read_region(1, 700_000, 1_500_000)
    print(f"  read_region(chr1, 0.7-1.5Mb): {region.shape[0]:,} SNPs × {region.shape[1]} features in {time.time()-t0:.3f}s")

    print(f"\n{'ALL PASS' if all_pass else 'SOME TESTS FAILED'}")

    import shutil
    shutil.rmtree(tmpdir)
