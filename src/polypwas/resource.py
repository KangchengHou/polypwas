"""Download and cache public example resources."""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

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
EXAMPLE_RELEASE_TAG = "angptl3-ldl-example"
EXAMPLE_RELEASE_BASE_URL = (
    "https://github.com/KangchengHou/polypwas/releases/download/angptl3-ldl-example"
)

EXAMPLE_PQTL_PATH = EXAMPLES_DIR / "angptl3.ma.gz"
EXAMPLE_GWAS_PATH = EXAMPLES_DIR / "ldl.ma.gz"
EXAMPLE_WGTS_PATH = EXAMPLES_DIR / "angptl3.wgts.gz"
EXAMPLE_GENE_INFO_PATH = EXAMPLES_DIR / "angptl3.gene.tsv"
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
                    gh_path, "release", "download", EXAMPLE_RELEASE_TAG,
                    "--pattern", asset_name,
                    "--dir", str(dest.parent),
                    "--repo", "KangchengHou/polypwas",
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


def prepare_example_resources(include_weights: bool = False) -> dict[str, Path]:
    """Ensure bundled example inputs and public resources are ready."""
    example_inputs = ensure_example_inputs(include_weights=include_weights)
    ldm_dir = ensure_hm3_ldm()
    if not EXAMPLE_GENE_INFO_PATH.exists():
        EXAMPLE_GENE_INFO_PATH.write_text(
            "ID\tCHROM\tSTART\tEND\n"
            f"ANGPTL3\t{EXAMPLE_GENE_COORDS['chrom']}\t"
            f"{EXAMPLE_GENE_COORDS['start']}\t{EXAMPLE_GENE_COORDS['end']}\n"
        )
    result = {
        "pqtl": example_inputs[EXAMPLE_PQTL_PATH.name],
        "gwas": example_inputs[EXAMPLE_GWAS_PATH.name],
        "ldm_dir": ldm_dir,
        "gene_info": EXAMPLE_GENE_INFO_PATH,
        "gene_coords": EXAMPLE_GENE_COORDS,
    }
    if include_weights and EXAMPLE_WGTS_PATH.name in example_inputs:
        result["weights"] = example_inputs[EXAMPLE_WGTS_PATH.name]
    return result
