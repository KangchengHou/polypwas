"""Public command-line interface for the demo workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_CONFIG, write_config
from .demo import (
    compute_demo_pwas,
    prepare_demo_resources,
    train_demo_weights,
)
from .sbayesrc import validate_runtime


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(prog="polypwas")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download-example",
        help="Download and extract the public demo resources",
    )
    download_parser.add_argument(
        "--include-weights",
        action="store_true",
        help="Also download pre-trained weights (skip the train step)",
    )
    download_parser.set_defaults(handler=handle_download_example)

    setup_parser = subparsers.add_parser(
        "setup",
        help="Write ~/.polypwas/config.yaml for external tool paths",
    )
    setup_parser.add_argument("--rscript", type=str)
    setup_parser.add_argument("--plink2", type=str)
    setup_parser.set_defaults(handler=handle_setup)

    train_parser = subparsers.add_parser(
        "train",
        help="Create demo SNP weights from an explicit pQTL input file",
    )
    train_parser.add_argument("--pqtl", type=Path, required=True)
    train_parser.add_argument("--ldm-dir", type=Path, required=True)
    train_parser.add_argument("--annot", type=Path)
    train_parser.add_argument("--threads", type=int, default=10)
    train_parser.add_argument("--out", type=Path, required=True)
    train_parser.set_defaults(handler=handle_train)

    assoc_parser = subparsers.add_parser(
        "assoc",
        help="Compute demo cis and trans PWAS Z-scores from explicit input files",
    )
    assoc_parser.add_argument("--weights", type=Path, required=True)
    assoc_parser.add_argument("--gwas", type=Path, required=True)
    assoc_parser.add_argument("--ldm-dir", type=Path, required=True)
    assoc_parser.add_argument("--gene-info", type=Path)
    assoc_parser.add_argument("--gene-chr", type=str)
    assoc_parser.add_argument("--gene-start", type=int)
    assoc_parser.add_argument("--gene-end", type=int)
    assoc_parser.set_defaults(handler=handle_assoc)

    return parser


def handle_download_example(args: argparse.Namespace) -> int:
    """Ensure the public example resources are present locally."""
    resources = prepare_demo_resources(include_weights=args.include_weights)
    print(f"Example pQTL: {resources['pqtl']}")
    print(f"Example GWAS: {resources['gwas']}")
    print(f"HM3 LD: {resources['ldm_dir']}")
    if "weights" in resources:
        print(f"Pre-trained weights: {resources['weights']}")
    coords = resources["gene_coords"]
    print(f"Demo gene coordinates: chr{coords['chrom']}:{coords['start']}-{coords['end']}")
    return 0


def _prompt_or_default(label: str, provided: str | None, default: str) -> str:
    """Return an explicit value or prompt the user with a default."""
    if provided is not None:
        return provided
    response = input(f"{label} [default: {default}]: ").strip()
    return response or default


def handle_setup(args: argparse.Namespace) -> int:
    """Write the polypwas config file."""
    rscript = _prompt_or_default("Rscript path", args.rscript, DEFAULT_CONFIG["Rscript"])
    validated_rscript = validate_runtime(rscript)
    config_path = write_config(rscript=rscript, plink2=args.plink2)
    print(f"Wrote config to {config_path}")
    print(f"Rscript: {validated_rscript}")
    if args.plink2 is not None:
        print(f"plink2: {args.plink2}")
    return 0


def handle_train(args: argparse.Namespace) -> int:
    """Run explicit SBayesRC-backed training."""
    validate_runtime()
    out_path = train_demo_weights(
        pqtl_path=args.pqtl,
        ldm_dir=args.ldm_dir,
        out_path=args.out,
        annot_path=args.annot,
        threads=args.threads,
    )
    print(f"Wrote demo weights to {out_path}")
    return 0


def handle_assoc(args: argparse.Namespace) -> int:
    """Compute and print demo cis/trans PWAS Z-scores."""
    cis_z, trans_z = compute_demo_pwas(
        weights_path=args.weights,
        gwas_path=args.gwas,
        ldm_dir=args.ldm_dir,
        gene_info_path=args.gene_info,
        gene_chr=args.gene_chr,
        gene_start=args.gene_start,
        gene_end=args.gene_end,
    )
    print(f"CIS_Z={cis_z:.6f}")
    print(f"TRANS_Z={trans_z:.6f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the public CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))
