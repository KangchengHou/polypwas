"""One-off builder for the HM3 ANGPTL3 + LDL public example dataset."""

from pathlib import Path

from polypwas.example_data import build_hm3_example_data


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "analyses" / "external"
OUT_DIR = ROOT / "data" / "examples"
PROTEIN_ID = "ANGPTL3:Q9Y5C1:OID20407:v1"


def main() -> None:
    bundle = build_hm3_example_data(
        protein_id=PROTEIN_ID,
        pqtl_path=EXTERNAL / "pqtl" / "sumstats" / "ukbsun" / f"{PROTEIN_ID}.ma.gz",
        gene_path=EXTERNAL / "pqtl" / "sumstats" / "ukbsun.gene.tsv",
        gwas_path=EXTERNAL / "gwas" / "price2_compiled" / "biochemistry_LDLdirect.ma",
        imputed_snp_info_path=EXTERNAL / "ldm" / "ukbEUR_Imputed" / "snp.info",
        hm3_snp_info_path=EXTERNAL / "ldm" / "ukbEUR_HM3" / "snp.info",
        out_dir=OUT_DIR,
    )
    print(f"Wrote {len(bundle.pqtl):,} HM3 pQTL rows to {OUT_DIR / 'angptl3.ma.gz'}")
    print(f"Wrote {len(bundle.gwas):,} HM3 GWAS rows to {OUT_DIR / 'ldl.ma.gz'}")


if __name__ == "__main__":
    main()
