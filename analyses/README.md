# Quick demo

Run cis/trans PWAS Z-scores for the first 10 (alphabetical) ukbsun proteins against the LDL-C GWAS, on both HM3 and imputed LD panels. Output is a TSV with `ID, CIS_Z, TRANS_Z`.

```bash
#!/usr/bin/env bash
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
```

# Datasets (inputs)

Data for polypwas lives under `analyses/datasets/`. We plan to release the per-cohort pQTL sumstats and SBayesRC weights upon publication. The directory structure is:

```
datasets/
├── pqtl-sumstats/   →  /n/groups/price/PQTLGWAS/pqtl-sumstats/
│   ├── {cohort}/{pid}.ma.gz       # per-protein sumstats: freq, b, se, N, r2 (rows aligned to LDM SNP order)
│   ├── {cohort}.gene.tsv          # ID, UNIPROT, CHROM, START, END, ASSAY, ENSEMBL, EXONS
│   ├── {cohort}.pqtl.tsv          # per-SNP boolean: pqtl_cis, pqtl_trans, pqtl_both (P < 5e-8)
│   └── {cohort}.pqtl_count.tsv    # per-SNP integer: cis_count, trans_count, total_count
│
├── pqtl-weights/    →  /n/groups/price/PQTLGWAS/pqtl-sbayesrc/
│   ├── {group}.parquet            # block-aligned (n_snp × n_protein) → polypwas.store.BlockWgt
│   └── {group}/{pid}.{par,log,AnnoJointProb,AnnoPerSnpHsqEnrichment}
│                                  # SBayesRC training artefacts (per-SNP .tsv.gz weights are backed up to /n/scratch/...)
│
├── sbayesrc/        →  /n/groups/price/UKBiobank/UKBPPP/DATA/
│   ├── ukbEUR_HM3/                # SBayesRC eigendecomposition LD (1.15M HM3 SNPs)
│   │   ├── block{1..591}.eigen.bin
│   │   ├── snp.info
│   │   ├── ldm.info
│   │   └── snplist.txt
│   ├── ukbEUR_Imputed/            # SBayesRC eigendecomposition LD (7.3M imputed SNPs)
│   │   ├── block{1..591}.eigen.bin
│   │   ├── snp.info
│   │   ├── ldm.info
│   │   └── snplist.txt
│   └── annot_baseline2.2.txt      # baseline-LD v2.2 annotations
│
├── gwas/
│   ├── price2/{trait}.stats.gz          # raw UKB BOLT-LMM output (32 traits, excl. PPP overlap)
│   ├── price2_compiled/{trait}.ma       # LDM-aligned .ma format
│   ├── pass_sumstats/{trait}.sumstats.gz # raw PASS consortium sumstats
│   ├── pass_compiled/{trait}.ma         # LDM-aligned .ma format
│   ├── trait_info.tsv                   # trait → cohort, N, h², category
│   ├── trait_values.tsv                 # individual-level trait values for UKB traits
│   ├── indep_gwas_traits.tsv            # ~independent trait subset (LDSC r² < 0.25)
│   ├── ldsc_rg.txt                      # LDSC genetic correlations
│   ├── ldsc_hsq.txt                     # LDSC heritabilities
│   ├── burden.tsv                       # rare-variant burden test results
│   └── pops.tsv                         # PoPS pathway scores
│
└── ukbppp/
    ├── protein.pheno                 # measured protein levels (~54K individuals × ~2,800 proteins)
    ├── protein.covar                 # covariates (age, sex, batch, PCs)
    ├── genotype_impute+acc/          # merged PGEN for the 62,856 imputation+accuracy individuals
    │   ├── merged.pgen
    │   ├── merged.pvar
    │   └── merged.psam
    ├── genotype_bgen/                # raw UKB BGEN (used to build the merged PGEN)
    ├── impute.indivlist              # 49,999 individuals used for SBayesRC weight training
    ├── acc.indivlist                 # 12,855 held-out individuals for prediction accuracy
    ├── acc+impute.indivlist          # combined 62,856 individual list
    └── unrelated_337K.txt            # unrelated EUR individuals (for QC)
```

## Cohorts

| Cohort | Description |
|--------|-------------|
| `ukbsun` | Sun et al. UKB-PPP official release |
| `ukb_linreg_0pc` | UKB-PPP linear regression, 0 protein PCs regressed out |
| `ukb_linreg_20pc` | UKB-PPP linear regression, 20 protein PCs regressed out |
| `decode` | deCODE plasma pQTLs (4,707 SeqIds) |
| `csf` | Western et al. CSF pQTLs (6,597 aptamers) |

## Weight groups

Weight files are named `{cohort}.{snp_set}.{annot}`, with `snp_set ∈ {hm3 (1.15M SNPs), imputed (7.3M SNPs)}` and `annot ∈ {none, baseline+cis, baseline+cis+pqtl}`. Availability per in-scope cohort:

| Cohort | `hm3.none` | `hm3.baseline+cis` | `hm3.baseline+cis+pqtl` | `imputed.none` | `imputed.baseline+cis` | `imputed.baseline+cis+pqtl` |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| `ukbsun`           | ✓ |   | ✓ | ✓ |   | ✓ |
| `ukb_linreg_0pc`   | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ukb_linreg_20pc`  | ✓ | ✓ | ✓ | parquet only |   | ✓ |
| `decode`           |   |   | ✓ |   |   | ✓ |
| `csf`              |   |   | ✓ |   |   | ✓ |

The primary group used by polypwas is `ukbsun.imputed.baseline+cis+pqtl`.

## Conventions

- **pQTL sumstats** arrive as per-protein `.ma.gz` files from the upstream pqtl repo with rows aligned to the LDM SNP order (columns `freq, b, se, N, r2`; `SNP/A1/A2` are implicit via the LDM `snp.info`). polypwas does not re-align them.
- **GWAS sumstats** are formatted to `.ma` (`SNP A1 A2 freq b se p N`) by `compile_*_sumstats.py` in *GWAS Compilation*; outputs land in `gwas/{price2,pass}_compiled/`.
- **LDM reference**: `ukbEUR_Imputed` is the primary LD panel (matches `ukbsun.imputed.baseline+cis+pqtl`); `ukbEUR_HM3` is used for the smaller HM3 weight set and for simulation. Eigendecomposition LD lives under `datasets/sbayesrc/`.
- **Individual-level UKB data** (genotype, protein, trait values) are restricted to the EUR unrelated subset and the imputation/accuracy split fixed in `*.indivlist` files (see `datasets/ukbppp/`).
- All paths are configurable via `~/.polypwas/config.yaml`; nothing should be hard-coded against `/n/groups/price/`.

---

# pQTL Compilation

Formatting pQTL sumstats, building pQTL annotations, and training SBayesRC weights are owned by the upstream pqtl repo at `/n/groups/price/PQTLGWAS/`. Their outputs are surfaced here as `datasets/pqtl-sumstats/` and `datasets/pqtl-weights/` (see above).

Two helper scripts live in this repo:

| Script | Output |
|--------|--------|
| `pqtl-compilation/convert_to_parquet.py` | `datasets/pqtl-weights/{group}.parquet` (block-aligned `BlockWgt`) |
| `pqtl-compilation/make_pqtl_count.py`    | `datasets/pqtl-sumstats/{cohort}.pqtl_count.tsv` |

---

# GWAS Compilation

Select independent traits and prepare GWAS sumstats from `datasets/gwas/`.

## 1. Select independent traits

Use LDSC genetic correlations (`ldsc_rg.txt`, `ldsc_hsq.txt`) to pick ~independent traits (r² < 0.25), prioritizing high heritability.

| Script | Source | Output |
|--------|--------|--------|
| `ukb_indep_traits.py` | UKB BOLT-LMM | `DATA/indep_traits.tsv` |
| `pass_indep_traits.py` | PASS consortium | `DATA/pass_indep_traits.tsv` |

## 2. Compile GWAS sumstats

Format to .ma aligned with LDM reference SNPs via SBayesRC munging.

| Script | Source | Output |
|--------|--------|--------|
| `compile_price2_sumstats.py` | UKB (excl PPP) | `DATA/PRICE2_GWAS/{trait}.ma` |
| `compile_pass_sumstats.py` | PASS consortium | `DATA/PASS_GWAS/{trait}.ma` |

---

# Protein Prediction

Compute polygenic scores from SBayesRC weights and evaluate prediction accuracy. Depends on: pqtl-compilation (weights), gwas-compilation (trait values).

## 3. Extract genotypes

Slice/merge `datasets/ukbppp/genotype_bgen/` into the merged PGEN restricted to the `acc+impute` indivlist.

| Script | Output |
|--------|--------|
| `extract_genotype.py` | `datasets/ukbppp/genotype_impute+acc/merged.{pgen,pvar,psam}` |

## 4. Compute polygenic scores

PLINK2 `--score` using SBayesRC weights, stratified by SNP subset (cis, trans, cisgenic, cisnongenic, cisexonic, cisnonexonic).

| Script | Output |
|--------|--------|
| `run_prediction.py` | `DATA/prediction/{group}.{subset}.parquet` |

## 5. Evaluate prediction accuracy

Correlate predicted vs measured protein levels in held-out UKB individuals (`acc.indivlist`).

| Script | Output |
|--------|--------|
| `eval_ukb_acc.py` | `DATA/predacc_stats.tsv` |

## 6. Variance explained in traits

Regress predicted protein PCs on 10 UKB traits, quantify cis/trans R².

| Script | Output |
|--------|--------|
| `run_variance_explained.py` | `DATA/variance-explained/{group}.{trait}.{n_pc}pc.json` |

---

# PWAS Analysis

Compute PWAS Z-scores, co-regulation matrices, and validate against independent evidence. Depends on: pqtl-compilation (weights, annotations), gwas-compilation (GWAS .ma files).

## 7. Compute PWAS covariances

For each protein × trait, compute Z-score numerators (weight · GWAS-z) stratified by cis/trans.

| Script | Output |
|--------|--------|
| `compute_pwas.py` | `DATA/pwas/{gwas_group}/{pqtl_group}.tsv.gz` |

## 8. Compute co-regulation matrices

Protein × protein covariance (w'Rw) for cis and trans components.

| Script | Output |
|--------|--------|
| `compute_coreg.py` | `DATA/coreg/{group}.{cis\|trans}.parquet` |

## 9. Compute variance components

Diagonal of co-regulation (w'Rw per protein) for Z-score denominators.

| Script | Output |
|--------|--------|
| `compute_var.py` | `DATA/var/{group}.{cis\|trans}.parquet` |

## 10. Compile PWAS results

Combine covariance + variance → Z-scores, apply co-regulation PC regression (n_pc=20), bin by signal strength.

| Script | Output |
|--------|--------|
| `compile_pwas_df.py` | `DATA/pwas/{gwas_group}.{pqtl_group}.tsv` (CIS_Z, TRANS_Z columns) |

## 11. Optimize PC regression

Tune number of co-regulation PCs to remove using genomic jackknife. Tests n_pc ∈ {0, 5, 10, 15, 20}.

| Script | Output |
|--------|--------|
| `optimize_regress_pc.py` | `DATA/optimize_regress_pc/{group}.n_pc{n}.tsv` |

## 12. Burden test & PoPS validation

Test whether PWAS-significant proteins are enriched for rare-variant burden hits (`burden.tsv`) and PoPS pathway scores (`pops.tsv`).

| Script | Output |
|--------|--------|
| `run_price_validation.py` | `DATA/price_validation/{group}.{tsv,enrichment.tsv,regression.tsv}` |

---

# Simulation

FPR/power simulations for PWAS method validation. Depends on the `ukbEUR_HM3` SBayesRC LDM at `datasets/sbayesrc/ukbEUR_HM3/`. Runs independently of GWAS/PWAS stages.

## 13. Run simulations

Simulate pQTL (SBayesRC MCMC) + GWAS with mediated/independent/correlated architectures, compute PWAS Z-scores across 100s of replicates.

| Script | Output |
|--------|--------|
| `simulate.py` | `DATA/sim/{params}.pwasz.txt`, `DATA/sim/{params}.par.tsv` |

Parameter grid:

| Panel | Varies | Purpose |
|-------|--------|---------|
| A | protein h² (4 × 1000 seeds) | FPR under null GWAS |
| B | protein h² (4 × 100 seeds) | Power with mediated effect |
| C | genetic correlation (4 × 100 seeds) | Tagging by shared genetics |
| D | independent h² (4 × 2 × 100 seeds) | FPR with non-mediated signal |
| Secondary | polygenicity | Sensitivity analysis |

File naming: `phsq={h2}|cor={cor}|indep={indep}|mediated={med}|seed={seed}[|prot_pcausal={p}|indep_pcausal={p}]`

---

# Paper

Manuscript pipeline: derive → tables → figures → statistics. Depends on all upstream stages.

All scripts must run in order:

```bash
uv run python derive.misc.py          # → derived/misc/
uv run python derive.prediction.py    # → derived/prediction/
uv run python derive.pwas.py          # → derived/pwas/
uv run python derive.validation.py    # → derived/validation/
uv run python derive.simulation.py    # → derived/simulation/
uv run python tables.py               # → tables/*.tsv.gz + tables.xlsx
uv run python figures.prediction.py   # → figures/prediction/
uv run python figures.pwas.py         # → figures/pwas/
uv run python figures.validation.py   # → figures/validation/
uv run python figures.simulation.py   # → figures/simulation/
uv run python report.stats.py         # → report.stats.txt
```

`utils.py` provides shared constants (`MAIN_GROUP`, `MAIN_TRAITS`, `CROSS_COHORTS`) and I/O helpers (`read_table()`, `read_derived()`, `write_derived()`).

```
paper/
├── derived/{misc,prediction,pwas,validation,simulation}/
├── tables/*.tsv.gz + tables.xlsx
├── figures/{prediction,pwas,validation,simulation}/*.png
└── report.stats.txt
```
