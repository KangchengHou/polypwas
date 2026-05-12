"""Public command-line interface for the polypwas workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import DEFAULT_CONFIG, write_config
from .resource import prepare_example_resources
from .sbayesrc import validate_runtime, build_sbayesrc_ma

logger = logging.getLogger("polypwas")


def _setup_logging(verbose: bool) -> None:
    if logger.handlers:
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False


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
        help="Train SNP weights from a pQTL input file using SBayesRC",
    )
    train_parser.add_argument("--pqtl", type=Path, required=True)
    train_parser.add_argument("--ldm-dir", type=Path, required=True)
    train_parser.add_argument("--annot", type=Path)
    train_parser.add_argument("--threads", type=int, default=10)
    train_parser.add_argument("--out", type=Path, required=True)
    train_parser.set_defaults(handler=handle_train)

    assoc_parser = subparsers.add_parser(
        "assoc",
        help="Compute cis and trans PWAS Z-scores (single or many proteins)",
    )
    assoc_parser.add_argument(
        "--weights",
        type=Path,
        nargs="+",
        required=True,
        help="One or more per-protein weight files (with BETA column), or a single .parquet BlockWgt",
    )
    assoc_parser.add_argument("--gwas", type=Path, required=True)
    assoc_parser.add_argument("--ldm-dir", type=Path, required=True)
    assoc_parser.add_argument(
        "--gene-info",
        type=Path,
        required=True,
        help="TSV with columns ID, CHROM, START, END (single weight file uses first row)",
    )
    assoc_parser.add_argument(
        "--out",
        type=Path,
        help="Output TSV path (required when computing >1 protein)",
    )
    assoc_parser.add_argument("--cis-window", type=float, default=1e6)
    assoc_parser.add_argument("--verbose", action="store_true", help="Show per-block progress bar")
    assoc_parser.set_defaults(handler=handle_assoc)

    return parser


def handle_download_example(args: argparse.Namespace) -> int:
    """Ensure the public example resources are present locally."""
    resources = prepare_example_resources(include_weights=args.include_weights)
    print(f"Example pQTL: {resources['pqtl']}")
    print(f"Example GWAS: {resources['gwas']}")
    print(f"HM3 LD: {resources['ldm_dir']}")
    print(f"Gene info: {resources['gene_info']}")
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
    """Train SBayesRC weights from a pQTL input file."""
    import gzip
    import shutil
    import tempfile
    from .sbayesrc import train as sbayesrc_train

    validate_runtime()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Preparing SBayesRC input...")
    ma_df = build_sbayesrc_ma(pqtl_path=args.pqtl, ldm_dir=args.ldm_dir)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ma_path = Path(tmp_dir) / "train.ma"
        sbayesrc_prefix = Path(tmp_dir) / "sbayesrc_weights"
        ma_df.to_csv(ma_path, sep="\t", index=False)
        print("Running SBayesRC training...")
        sbayesrc_train(
            ma_path=str(ma_path),
            ldm_dir=str(args.ldm_dir),
            annot_path=None if args.annot is None else str(args.annot),
            out_prefix=str(sbayesrc_prefix),
            threads=args.threads,
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

    print(f"Wrote weights to {out_path}")
    return 0


def handle_assoc(args: argparse.Namespace) -> int:
    """Compute cis/trans PWAS Z-scores for one or many proteins."""
    import pandas as pd
    from .ld import BlockLDM
    from .pwas import compute_pwas_z
    from .store import BlockWgt

    weights_paths = list(args.weights)
    is_parquet = len(weights_paths) == 1 and weights_paths[0].suffix == ".parquet"
    single_file = len(weights_paths) == 1 and not is_parquet
    to_stdout = single_file and args.out is None

    if not to_stdout:
        if args.out is None:
            raise SystemExit("--out is required for multi-protein assoc")
        _setup_logging(args.verbose)

    logger.info("Loading LDM from %s", args.ldm_dir)
    ldm = BlockLDM(str(args.ldm_dir))
    snp_info = ldm.snp_info
    logger.info("LDM: %d SNPs across %d blocks", len(snp_info), ldm.n_block)

    logger.info("Loading gene info from %s", args.gene_info)
    gene_info = pd.read_csv(args.gene_info, sep="\t", index_col="ID")
    if to_stdout:
        gene_info = gene_info.iloc[:1]
    logger.info("Gene info: %d proteins", len(gene_info))

    logger.info("Loading GWAS from %s", args.gwas)
    gwas = pd.read_csv(args.gwas, sep="\t", index_col="SNP")
    gwas_z = (gwas.loc[snp_info.index, "b"] / gwas.loc[snp_info.index, "se"]).astype(float)
    logger.info("GWAS Z aligned: %d/%d SNPs", gwas_z.notna().sum(), len(snp_info))

    if is_parquet:
        logger.info("Opening weights parquet %s", weights_paths[0])
        weights = BlockWgt(str(weights_paths[0]))
        available = set(weights.columns)
        pids = [p for p in gene_info.index if p in available]
        if not pids:
            raise SystemExit("No protein IDs in --gene-info match parquet columns")
        dropped = len(gene_info) - len(pids)
        if dropped:
            logger.warning("Dropped %d gene-info IDs not in parquet", dropped)
        gene_info = gene_info.loc[pids]
        logger.info(
            "Parquet: %d proteins selected; file has %d columns × %d SNPs",
            len(pids), weights.n_features, weights.n_snp,
        )
    else:
        logger.info("Reading %d per-protein weight files", len(weights_paths))
        cols = {}
        for path in weights_paths:
            pid = path.name
            for suf in (".tsv.gz", ".tsv", ".gz"):
                if pid.endswith(suf):
                    pid = pid[: -len(suf)]
                    break
            df = pd.read_csv(path, sep="\t")
            if "BETA" not in df.columns:
                raise SystemExit(f"{path}: missing BETA column")
            if "SNP" in df.columns:
                col = df.set_index("SNP")["BETA"].reindex(snp_info.index).fillna(0.0)
            elif len(df.columns) == 1:
                if len(df) != len(snp_info):
                    raise SystemExit(
                        f"{path}: BETA-only file has {len(df)} rows, LDM has {len(snp_info)}"
                    )
                col = pd.Series(df["BETA"].to_numpy(), index=snp_info.index)
            else:
                raise SystemExit(f"{path}: need [SNP, BETA] or BETA-only")
            cols[pid] = col
        weights = pd.DataFrame(cols)
        missing = [p for p in gene_info.index if p not in weights.columns]
        if missing:
            raise SystemExit(f"--gene-info has IDs without weight files: {missing[:5]}")
        weights = weights[list(gene_info.index)]

    logger.info("Computing PWAS Z across %d blocks", ldm.n_block)
    result = compute_pwas_z(
        weights=weights,
        gwas_z=gwas_z,
        ldm=ldm,
        gene_info=gene_info,
        cis_window=args.cis_window,
        verbose=args.verbose,
    )

    if to_stdout:
        print(f"CIS_Z={result['CIS_Z'].iloc[0]:.6f}")
        print(f"TRANS_Z={result['TRANS_Z'].iloc[0]:.6f}")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, sep="\t", index=False, float_format="%.6f")
        logger.info("Wrote %d protein Z-scores to %s", len(result), args.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the public CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))
