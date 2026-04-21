import pandas as pd

from polypwas.example_data import make_hm3_example_bundle


def test_make_hm3_example_bundle_attaches_snp_ids_and_filters_to_hm3():
    imputed_snp_info = pd.DataFrame(
        {
            "ID": ["rs1", "rs2", "rs3", "rs4"],
            "Chrom": [1, 1, 1, 1],
            "PhysPos": [100, 200, 300, 400],
        }
    )
    hm3_snp_info = pd.DataFrame({"ID": ["rs2", "rs4"]})

    pqtl_sumstats = pd.DataFrame(
        {
            "freq": [0.1, 0.2, 0.3, 0.4],
            "b": [1.0, 2.0, 3.0, 4.0],
            "se": [0.1, 0.2, 0.3, 0.4],
            "N": [1000, 1000, 1000, 1000],
            "r2": [1.0, 1.0, 1.0, 1.0],
        }
    )
    gwas_sumstats = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3", "rs4"],
            "A1": ["A", "C", "G", "T"],
            "A2": ["G", "T", "A", "C"],
            "freq": [0.1, 0.2, 0.3, 0.4],
            "b": [0.1, 0.2, 0.3, 0.4],
            "se": [0.01, 0.02, 0.03, 0.04],
            "p": [0.5, 0.4, 0.3, 0.2],
            "N": [2000, 2000, 2000, 2000],
            "r2": [1.0, 1.0, 1.0, 1.0],
        }
    )
    gene_info = pd.DataFrame(
        {
            "UNIPROT": ["Q9Y5C1"],
            "CHROM": [1],
            "START": [63063158],
            "END": [63071830],
            "GENE": ["ANGPTL3"],
            "ENSEMBL": ["ENSG00000132855"],
            "EXONS": ["63063238-63063732"],
        },
        index=["ANGPTL3:Q9Y5C1:OID20407:v1"],
    )

    bundle = make_hm3_example_bundle(
        protein_id="ANGPTL3:Q9Y5C1:OID20407:v1",
        pqtl_sumstats=pqtl_sumstats,
        gwas_sumstats=gwas_sumstats,
        gene_info=gene_info,
        imputed_snp_info=imputed_snp_info,
        hm3_snp_info=hm3_snp_info,
    )

    assert bundle.pqtl["SNP"].tolist() == ["rs2", "rs4"]
    assert bundle.pqtl["b"].tolist() == [2.0, 4.0]
    assert bundle.gwas["SNP"].tolist() == ["rs2", "rs4"]
    assert list(bundle.gene.index) == ["ANGPTL3:Q9Y5C1:OID20407:v1"]
