"""Public demo workflow helpers for bundled example data."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
import scipy.stats
import tempfile
import gzip
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .ld import BlockLDM
from .sbayesrc import train as sbayesrc_train


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
EXAMPLES_DIR = DATA_DIR / "examples"
LDM_ROOT = DATA_DIR / "ldm"
HM3_DIR = LDM_ROOT / "ukbEUR_HM3"

HM3_ZIP_URL = (
    "https://gctbhub.cloud.edu.au/data/SBayesRC/resources/v2.0/LD/HapMap3/ukbEUR_HM3.zip"
)
HM3_ZIP_PATH = DOWNLOADS_DIR / "ukbEUR_HM3.zip"
EXAMPLE_RELEASE_BASE_URL = (
    "https://github.com/KangchengHou/polypwas/releases/download/angptl3-ldl-example"
)
EXAMPLE_RELEASE_TAG = "angptl3-ldl-example"

EXAMPLE_PQTL_PATH = EXAMPLES_DIR / "angptl3.ma.gz"
EXAMPLE_GWAS_PATH = EXAMPLES_DIR / "ldl.ma.gz"
EXAMPLE_WGTS_PATH = EXAMPLES_DIR / "angptl3.wgts.gz"
EXAMPLE_ASSET_URLS = {
    EXAMPLE_PQTL_PATH.name: f"{EXAMPLE_RELEASE_BASE_URL}/{EXAMPLE_PQTL_PATH.name}",
    EXAMPLE_GWAS_PATH.name: f"{EXAMPLE_RELEASE_BASE_URL}/{EXAMPLE_GWAS_PATH.name}",
    EXAMPLE_WGTS_PATH.name: f"{EXAMPLE_RELEASE_BASE_URL}/{EXAMPLE_WGTS_PATH.name}",
}
EXAMPLE_GENE_COORDS = {
    "chrom": 1,
    "start": 63063158,
    "end": 63071830,
}

HM3_REQUIRED_FILES = ["snp.info", "ldm.info", "block1.eigen.bin"]


def download_file(url: str, dest: str | Path) -> Path:
    """Download a file if missing."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        return dest

    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    tmp_path.replace(dest)
    return dest


def download_release_asset(asset_name: str, dest: str | Path) -> Path:
    """Download an example asset from the configured GitHub release."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        return dest

    gh_path = shutil.which("gh")
    if gh_path is not None:
        try:
            subprocess.run(
                [
                    gh_path,
                    "release",
                    "download",
                    EXAMPLE_RELEASE_TAG,
                    "--pattern",
                    asset_name,
                    "--dir",
                    str(dest.parent),
                    "--repo",
                    "KangchengHou/polypwas",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if dest.exists():
                return dest
        except subprocess.CalledProcessError:
            pass

    return download_file(EXAMPLE_ASSET_URLS[asset_name], dest)


def extract_zip(archive_path: str | Path, dest_dir: str | Path) -> None:
    """Extract a ZIP archive with validation and directory creation."""
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Archive integrity failure in member: {bad_member}")
        archive.extractall(dest_dir)


def ensure_example_inputs(include_weights: bool = False) -> dict[str, Path]:
    """Download the public example inputs if they are missing."""
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    resolved = {}
    paths = [EXAMPLE_PQTL_PATH, EXAMPLE_GWAS_PATH]
    if include_weights:
        paths.append(EXAMPLE_WGTS_PATH)
    for path in paths:
        if not path.exists():
            download_release_asset(path.name, path)
        resolved[path.name] = path
    return resolved


def ensure_hm3_ldm() -> Path:
    """Ensure the HM3 LD resource is downloaded and extracted."""
    if not all((HM3_DIR / name).exists() for name in HM3_REQUIRED_FILES):
        download_file(HM3_ZIP_URL, HM3_ZIP_PATH)
        extract_zip(HM3_ZIP_PATH, LDM_ROOT)

    missing = [name for name in HM3_REQUIRED_FILES if not (HM3_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"HM3 extraction incomplete under {HM3_DIR}: missing {', '.join(missing)}"
        )
    return HM3_DIR


def prepare_demo_resources(include_weights: bool = False) -> dict[str, Path]:
    """Ensure bundled example inputs and public resources are ready."""
    example_inputs = ensure_example_inputs(include_weights=include_weights)
    ldm_dir = ensure_hm3_ldm()
    result = {
        "pqtl": example_inputs[EXAMPLE_PQTL_PATH.name],
        "gwas": example_inputs[EXAMPLE_GWAS_PATH.name],
        "ldm_dir": ldm_dir,
        "gene_coords": EXAMPLE_GENE_COORDS,
    }
    if include_weights and EXAMPLE_WGTS_PATH.name in example_inputs:
        result["weights"] = example_inputs[EXAMPLE_WGTS_PATH.name]
    return result


def build_sbayesrc_ma(pqtl_path: str | Path, ldm_dir: str | Path) -> pd.DataFrame:
    """Build an SBayesRC-compatible .ma table from explicit pQTL input."""
    pqtl_df = pd.read_csv(pqtl_path, sep="\t")

    required = {"SNP", "freq", "b", "se", "N"}
    missing = required - set(pqtl_df.columns)
    if missing:
        raise ValueError(f"pQTL input missing required columns: {sorted(missing)}")

    if {"A1", "A2"}.issubset(pqtl_df.columns):
        ma_df = pqtl_df.copy()
    else:
        snp_info = pd.read_csv(Path(ldm_dir) / "snp.info", sep="\t", usecols=["ID", "A1", "A2"])
        ma_df = pqtl_df.merge(
            snp_info, left_on="SNP", right_on="ID", how="left", validate="many_to_one"
        )
        ma_df = ma_df.drop(columns=["ID"])

    if ma_df[["A1", "A2"]].isna().any().any():
        raise ValueError("Could not attach A1/A2 alleles for all SNPs from ldm_dir/snp.info")

    if "p" not in ma_df.columns:
        zscore = (ma_df["b"] / ma_df["se"]).astype(float)
        ma_df["p"] = scipy.stats.norm.sf(np.abs(zscore)) * 2

    return ma_df[["SNP", "A1", "A2", "freq", "b", "se", "p", "N"]]


def train_demo_weights(
    pqtl_path: str | Path,
    ldm_dir: str | Path,
    out_path: str | Path,
    annot_path: str | Path | None = None,
    threads: int = 10,
) -> Path:
    """Run SBayesRC training for an explicit pQTL input file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Preparing SBayesRC input...")
    ma_df = build_sbayesrc_ma(pqtl_path=pqtl_path, ldm_dir=ldm_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        ma_path = Path(tmp_dir) / "train.ma"
        sbayesrc_prefix = Path(tmp_dir) / "sbayesrc_weights"
        ma_df.to_csv(ma_path, sep="\t", index=False)
        print("Running SBayesRC training...")
        sbayesrc_train(
            ma_path=str(ma_path),
            ldm_dir=str(ldm_dir),
            annot_path=None if annot_path is None else str(annot_path),
            out_prefix=str(sbayesrc_prefix),
            threads=threads,
        )
        trained_weights_path = Path(f"{sbayesrc_prefix}.txt")
        if not trained_weights_path.exists():
            raise FileNotFoundError(
                f"Expected SBayesRC weights file not found: {trained_weights_path}"
            )
        if out_path.suffix == ".gz":
            with trained_weights_path.open("rb") as src, gzip.open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copyfile(trained_weights_path, out_path)

    return out_path


def compute_demo_pwas(
    weights_path: str | Path,
    gwas_path: str | Path,
    ldm_dir: str | Path,
    gene_info_path: str | Path | None = None,
    gene_chr: str | int | None = None,
    gene_start: int | None = None,
    gene_end: int | None = None,
    cis_window: float = 1e6,
) -> tuple[float, float]:
    """Compute demo cis/trans PWAS Z-scores from block-eigen LD (single protein)."""
    ldm = BlockLDM(str(ldm_dir))
    snp_info = ldm.snp_info

    weights_df = pd.read_csv(weights_path, sep="\t")
    if {"SNP", "BETA"}.issubset(weights_df.columns):
        weights = weights_df.set_index("SNP")["BETA"].reindex(snp_info.index).fillna(0.0)
    elif "BETA" in weights_df.columns and len(weights_df.columns) == 1:
        if len(weights_df) != len(snp_info):
            raise ValueError(
                f"BETA-only weights file has {len(weights_df)} rows but LD panel has {len(snp_info)} SNPs"
            )
        weights = pd.Series(weights_df["BETA"].to_numpy(), index=snp_info.index)
    else:
        raise ValueError("Weights file must contain either [SNP, BETA] or a single BETA column")

    gwas = pd.read_csv(gwas_path, sep="\t", index_col="SNP")
    gwas_z = (gwas.loc[snp_info.index, "b"] / gwas.loc[snp_info.index, "se"]).astype(float)
    if gene_info_path is not None:
        gene = pd.read_csv(gene_info_path, sep="\t", index_col="ID")
        chrom, start, end = gene.iloc[0][["CHROM", "START", "END"]]
    else:
        if gene_chr is None or gene_start is None or gene_end is None:
            raise ValueError(
                "Provide either --gene-info or all of --gene-chr, --gene-start, and --gene-end"
            )
        chrom, start, end = gene_chr, gene_start, gene_end

    chrom_dtype = snp_info["Chrom"].dtype
    if pd.api.types.is_numeric_dtype(chrom_dtype):
        chrom = pd.Series([chrom]).astype(chrom_dtype)[0]
    else:
        chrom = str(chrom)

    scaled_weights = weights * np.sqrt(2 * snp_info["A1Freq"] * (1 - snp_info["A1Freq"]))
    cis_mask = (snp_info["Chrom"] == chrom) & (
        snp_info["PhysPos"].between(start - cis_window, end + cis_window)
    )
    trans_mask = ~cis_mask

    cis_numer = 0.0
    trans_numer = 0.0
    cis_denom = 0.0
    trans_denom = 0.0

    for block_idx, block_info in snp_info.groupby("Block"):
        block_snps = block_info.index
        block_gwas = gwas_z.loc[block_snps]
        cis_weights = scaled_weights.loc[block_snps] * cis_mask.loc[block_snps]
        trans_weights = scaled_weights.loc[block_snps] * trans_mask.loc[block_snps]

        cis_numer += float(cis_weights @ block_gwas)
        trans_numer += float(trans_weights @ block_gwas)
        cis_denom += float(ldm.block_cov(int(block_idx), cis_weights))
        trans_denom += float(ldm.block_cov(int(block_idx), trans_weights))

    cis_z = cis_numer / np.sqrt(max(cis_denom, 1e-30))
    trans_z = trans_numer / np.sqrt(max(trans_denom, 1e-30))
    return float(cis_z), float(trans_z)
