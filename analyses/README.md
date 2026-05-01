# Datasets (inputs)

All upstream stages assume the following inputs are staged under `datasets/` (or symlinked from external sources, e.g. `/n/groups/price/`). Nothing in `analyses/` will run until these are in place.

```
datasets/
├── pqtl/
│   ├── ukbsun/{pid}.tsv.gz           # UKB-PPP raw pQTL sumstats (~2,923 proteins)
│   ├── decode/{pid}.tsv.gz           # deCODE raw pQTL sumstats (~5,000 proteins)
│   ├── csf/{pid}.tsv.gz              # CSF raw pQTL sumstats (~6,000 aptamers)
│   ├── {dataset}.gene.tsv            # protein → ENSG, TSS/TES, sample size, EXONS
│   └── {dataset}.sample.tsv          # per-protein N (if not in gene.tsv)
│
├── sbayesrc/
│   ├── ukbEUR_HM3_ldm/               # SBayesRC eigendecomposition LD (HM3 SNPs)
│   │   ├── block{1..591}.eigen.bin
│   │   └── snp.info
│   ├── ukbEUR_Imputed_ldm/           # SBayesRC eigendecomposition LD (imputed SNPs)
│   │   ├── block{1..591}.eigen.bin
│   │   └── snp.info
│   └── annot_baseline2.2.txt         # baseline-LD v2.2 annotations for SBayesRC
│
├── gwas/
│   ├── price2/{trait}.sumstats.gz    # 32 UKB BOLT-LMM traits (excluding PPP overlap)
│   ├── pass/{trait}.sumstats.gz      # 56 PASS consortium traits
│   ├── trait_info.tsv                # trait → cohort, N, h², category
│   ├── trait_values.tsv              # individual-level trait values for UKB traits
│   ├── ldsc_rg.txt                   # LDSC genetic correlations (for trait independence)
│   ├── ldsc_hsq.txt                  # LDSC heritabilities
│   ├── burden.tsv                    # rare-variant burden test results (for validation)
│   └── pops.tsv                      # PoPS pathway scores (for validation)
│
└── ukbppp/
    ├── protein.pheno                 # measured protein levels (~54K individuals × ~2,923 proteins)
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

Conventions:

- **pQTL sumstats** are aligned to LDM reference SNPs in stage 2 (`process_pqtl.py`); raw inputs may use any column scheme as long as it carries SNP, A1, A2, BETA, SE, P, N.
- **GWAS sumstats** are formatted to `.ma` (`SNP A1 A2 freq b se p N`) in stage 6 (`compile_*_sumstats.py`).
- **LDM reference**: `ukbEUR_HM3` is the default LD panel; `ukbEUR_Imputed` is used for the main imputed weight set (`ukbsun.imputed.baseline+cis+pqtl`).
- **Individual-level UKB data** (genotype, protein, trait values) are restricted to the EUR unrelated subset and the imputation/accuracy split fixed in `*.indivlist` files.
- All paths are configurable via `~/.polypwas/config.yaml`; nothing should be hard-coded against `/n/groups/price/`.

---

# pQTL Compilation

Format pQTL sumstats and train protein prediction weights for each cohort in `datasets/pqtl/`.

## 1. Format & impute summary statistics

For each protein, align raw sumstats to LDM reference SNPs (allele matching/flipping), then run SBayesRC tidy + impute.

| Script | Output |
|--------|--------|
| `process_pqtl.py` | `DATA/sumstats/{dataset}/{pid}.ma.gz` |
| `format_ukb.py` | `DATA/sumstats/ukb_linreg_{n_pc}pc/{pid}.ma.gz` (pre-computed) |

## 2. Build pQTL annotations

Count genome-wide-significant (P < 5e-8) cis and trans pQTLs per SNP across all proteins.

Output: `DATA/sumstats/{dataset}.pqtl.tsv` (per-SNP cis/trans/both counts)

## 3. Train SBayesRC weights

For each protein × cohort, train SBayesRC with functional annotations.

| Script | Output |
|--------|--------|
| `train_sbayesrc.py` | `DATA/sbayesrc/{dataset}.{ldm}.{annot}/{pid}.tsv.gz` |

Annotation models: `baseline+cis`, `baseline+cis+pqtl`. LD panels: HM3, Imputed. Runtime: ~720 min/protein via submitit.

---

# GWAS Compilation

Select independent traits and prepare GWAS sumstats from `datasets/gwas/`.

## 4. Select independent traits

Use LDSC genetic correlations (`ldsc_rg.txt`, `ldsc_hsq.txt`) to pick ~independent traits (r² < 0.25), prioritizing high heritability.

| Script | Source | Output |
|--------|--------|--------|
| `ukb_indep_traits.py` | UKB BOLT-LMM | `DATA/indep_traits.tsv` |
| `pass_indep_traits.py` | PASS consortium | `DATA/pass_indep_traits.tsv` |

## 5. Compile GWAS sumstats

Format to .ma aligned with LDM reference SNPs via SBayesRC munging.

| Script | Source | Output |
|--------|--------|--------|
| `compile_price2_sumstats.py` | UKB (excl PPP) | `DATA/PRICE2_GWAS/{trait}.ma` |
| `compile_pass_sumstats.py` | PASS consortium | `DATA/PASS_GWAS/{trait}.ma` |

---

# Protein Prediction

Compute polygenic scores from SBayesRC weights and evaluate prediction accuracy. Depends on: pqtl-compilation (weights), gwas-compilation (trait values).

## 6. Extract genotypes

Slice/merge `datasets/ukbppp/genotype_bgen/` into the merged PGEN restricted to the `acc+impute` indivlist.

| Script | Output |
|--------|--------|
| `extract_genotype.py` | `datasets/ukbppp/genotype_impute+acc/merged.{pgen,pvar,psam}` |

## 7. Compute polygenic scores

PLINK2 `--score` using SBayesRC weights, stratified by SNP subset (cis, trans, cisgenic, cisnongenic, cisexonic, cisnonexonic).

| Script | Output |
|--------|--------|
| `run_prediction.py` | `DATA/prediction/{group}.{subset}.parquet` |

## 8. Evaluate prediction accuracy

Correlate predicted vs measured protein levels in held-out UKB individuals (`acc.indivlist`).

| Script | Output |
|--------|--------|
| `eval_ukb_acc.py` | `DATA/predacc_stats.tsv` |

## 9. Variance explained in traits

Regress predicted protein PCs on 10 UKB traits, quantify cis/trans R².

| Script | Output |
|--------|--------|
| `run_variance_explained.py` | `DATA/variance-explained/{group}.{trait}.{n_pc}pc.json` |

---

# PWAS Analysis

Compute PWAS Z-scores, co-regulation matrices, and validate against independent evidence. Depends on: pqtl-compilation (weights, annotations), gwas-compilation (GWAS .ma files).

## 10. Compute PWAS covariances

For each protein × trait, compute Z-score numerators (weight · GWAS-z) stratified by cis/trans.

| Script | Output |
|--------|--------|
| `compute_pwas.py` | `DATA/pwas/{gwas_group}/{pqtl_group}.tsv.gz` |

## 11. Compute co-regulation matrices

Protein × protein covariance (w'Rw) for cis and trans components.

| Script | Output |
|--------|--------|
| `compute_coreg.py` | `DATA/coreg/{group}.{cis\|trans}.parquet` |

## 12. Compute variance components

Diagonal of co-regulation (w'Rw per protein) for Z-score denominators.

| Script | Output |
|--------|--------|
| `compute_var.py` | `DATA/var/{group}.{cis\|trans}.parquet` |

## 13. Compile PWAS results

Combine covariance + variance → Z-scores, apply co-regulation PC regression (n_pc=20), bin by signal strength.

| Script | Output |
|--------|--------|
| `compile_pwas_df.py` | `DATA/pwas/{gwas_group}.{pqtl_group}.tsv` (CIS_Z, TRANS_Z columns) |

## 14. Optimize PC regression

Tune number of co-regulation PCs to remove using genomic jackknife. Tests n_pc ∈ {0, 5, 10, 15, 20}.

| Script | Output |
|--------|--------|
| `optimize_regress_pc.py` | `DATA/optimize_regress_pc/{group}.n_pc{n}.tsv` |

## 15. Burden test & PoPS validation

Test whether PWAS-significant proteins are enriched for rare-variant burden hits (`burden.tsv`) and PoPS pathway scores (`pops.tsv`).

| Script | Output |
|--------|--------|
| `run_price_validation.py` | `DATA/price_validation/{group}.{tsv,enrichment.tsv,regression.tsv}` |

---

# Simulation

FPR/power simulations for PWAS method validation. Depends on: `datasets/sbayesrc/ukbEUR_HM3_ldm/`. Runs independently of GWAS/PWAS stages.

## 16. Run simulations

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
