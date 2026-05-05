#!/usr/bin/env bash
# Toy run: cis/trans PWAS Z-scores for first 10 ukbsun proteins vs LDL-C GWAS,
# using HM3 / Imputed LDM and the ukbsun.{hm3,imputed}.baseline+cis+pqtl parquets.
set -euo pipefail

ROOT="/n/groups/price/kangcheng/projects/polypwas"
DATA="$ROOT/analyses/datasets"
OUT="$(mktemp -d -t toy_assoc.XXXXXX)"
GENE_INFO="$OUT/gene_info.tsv"
echo "output dir: $OUT"

head -n 11 "$DATA/pqtl-sumstats/ukbsun.gene.tsv" > "$GENE_INFO"

uv run --project "$ROOT" polypwas assoc --verbose \
    --weights   "$DATA/pqtl-weights/ukbsun.hm3.baseline+cis+pqtl.parquet" \
    --gwas      "$DATA/gwas/price2_compiled/biochemistry_LDLdirect.ma" \
    --ldm-dir   "$DATA/sbayesrc/ukbEUR_HM3" \
    --gene-info "$GENE_INFO" \
    --out       "$OUT/pwas_z.hm3.tsv"

uv run --project "$ROOT" polypwas assoc --verbose \
    --weights   "$DATA/pqtl-weights/ukbsun.imputed.baseline+cis+pqtl.parquet" \
    --gwas      "$DATA/gwas/price2_compiled/biochemistry_LDLdirect.ma" \
    --ldm-dir   "$DATA/sbayesrc/ukbEUR_Imputed" \
    --gene-info "$GENE_INFO" \
    --out       "$OUT/pwas_z.imputed.tsv"
