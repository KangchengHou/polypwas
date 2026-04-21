"""Helpers for building small public example datasets."""

from __future__ import annotations

import argparse
import gzip
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ExampleDataBundle:
    """HM3-filtered example inputs for one protein and one trait."""

    pqtl: pd.DataFrame
    gwas: pd.DataFrame
    gene: pd.DataFrame


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a TSV or gzipped TSV into a DataFrame."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f, sep="\t", **kwargs)
    return pd.read_csv(path, sep="\t", **kwargs)


def attach_snp_ids(sumstats: pd.DataFrame, snp_info: pd.DataFrame) -> pd.DataFrame:
    """Attach SNP IDs to LD-aligned summary statistics by row order."""
    if len(sumstats) != len(snp_info):
        raise ValueError(
            f"Row count mismatch: sumstats has {len(sumstats)} rows, "
            f"snp.info has {len(snp_info)} rows"
        )

    result = sumstats.copy()
    result.insert(0, "SNP", snp_info["ID"].to_numpy())
    return result


def make_hm3_example_bundle(
    *,
    protein_id: str,
    pqtl_sumstats: pd.DataFrame,
    gwas_sumstats: pd.DataFrame,
    gene_info: pd.DataFrame,
    imputed_snp_info: pd.DataFrame,
    hm3_snp_info: pd.DataFrame,
) -> ExampleDataBundle:
    """Build an HM3-only example bundle from imputed-aligned source tables."""
    pqtl_with_ids = attach_snp_ids(pqtl_sumstats, imputed_snp_info)
    hm3_ids = set(hm3_snp_info["ID"])

    pqtl_hm3 = pqtl_with_ids[pqtl_with_ids["SNP"].isin(hm3_ids)].reset_index(drop=True)
    gwas_hm3 = gwas_sumstats[gwas_sumstats["SNP"].isin(hm3_ids)].reset_index(drop=True)
    gene_row = gene_info.loc[[protein_id]].copy()

    return ExampleDataBundle(pqtl=pqtl_hm3, gwas=gwas_hm3, gene=gene_row)


def write_example_bundle(bundle: ExampleDataBundle, out_dir: str | Path) -> None:
    """Write example tables to a directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle.pqtl.to_csv(out_dir / "angptl3.ma.gz", sep="\t", index=False, compression="gzip")
    bundle.gwas.to_csv(out_dir / "ldl.ma.gz", sep="\t", index=False, compression="gzip")


def build_hm3_example_data(
    *,
    protein_id: str,
    pqtl_path: str | Path,
    gene_path: str | Path,
    gwas_path: str | Path,
    imputed_snp_info_path: str | Path,
    hm3_snp_info_path: str | Path,
    out_dir: str | Path,
) -> ExampleDataBundle:
    """Load source files, build HM3-filtered example data, and write outputs."""
    pqtl_sumstats = read_table(pqtl_path)
    gwas_sumstats = read_table(gwas_path)
    gene_info = read_table(gene_path, index_col=0)
    imputed_snp_info = read_table(imputed_snp_info_path)
    hm3_snp_info = read_table(hm3_snp_info_path)

    bundle = make_hm3_example_bundle(
        protein_id=protein_id,
        pqtl_sumstats=pqtl_sumstats,
        gwas_sumstats=gwas_sumstats,
        gene_info=gene_info,
        imputed_snp_info=imputed_snp_info,
        hm3_snp_info=hm3_snp_info,
    )
    write_example_bundle(bundle, out_dir)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HM3-only public example data.")
    parser.add_argument("--protein-id", required=True)
    parser.add_argument("--pqtl", required=True, help="Imputed-aligned pQTL .ma(.gz) path")
    parser.add_argument("--gene", required=True, help="Gene metadata TSV path")
    parser.add_argument("--gwas", required=True, help="GWAS .ma path")
    parser.add_argument("--imputed-snp-info", required=True, help="Imputed ldm/snp.info path")
    parser.add_argument("--hm3-snp-info", required=True, help="HM3 ldm/snp.info path")
    parser.add_argument("--out-dir", required=True, help="Output directory for example files")
    args = parser.parse_args()

    bundle = build_hm3_example_data(
        protein_id=args.protein_id,
        pqtl_path=args.pqtl,
        gene_path=args.gene,
        gwas_path=args.gwas,
        imputed_snp_info_path=args.imputed_snp_info,
        hm3_snp_info_path=args.hm3_snp_info,
        out_dir=args.out_dir,
    )
    print(f"Wrote {len(bundle.pqtl):,} HM3 pQTL rows to {Path(args.out_dir) / 'angptl3.ma.gz'}")
    print(f"Wrote {len(bundle.gwas):,} HM3 GWAS rows to {Path(args.out_dir) / 'ldl.ma.gz'}")


if __name__ == "__main__":
    main()
