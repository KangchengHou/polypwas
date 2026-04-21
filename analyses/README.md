# pQTL Compilation

Acquire, format, and train protein prediction weights from pQTL summary statistics across three cohorts: UKB-PPP (~2,923 proteins, ~54K individuals), deCODE (~5,000 proteins, ~35K individuals), CSF (~6,000 aptamers, ~3K individuals).

## 1. Download metadata

Retrieve protein lists, sample sizes, and genomic coordinates (TSS/TES + exons) from each source via Ensembl biomaRt.

| Script | Cohort | Output |
|--------|--------|--------|
| `csf_info.R` | CSF | `DATA/sumstats/csf.gene.tsv` |
| `decode_info.R` | deCODE | `DATA/sumstats/decode.gene.tsv` |
| `ukbsun_info.R` | UKB-PPP | `DATA/sumstats/ukbsun.gene.tsv` |
| `get_exon_info.R` | all | Adds EXONS column to gene.tsv files |

## 2. Format & impute summary statistics

For each protein, align raw sumstats to LDM reference SNPs (allele matching/flipping), then run SBayesRC tidy + impute.

| Script | Output |
|--------|--------|
| `process_pqtl.py` | `DATA/sumstats/{dataset}/{pid}.ma.gz` |
| `format_ukb.py` | `DATA/sumstats/ukb_linreg_{n_pc}pc/{pid}.ma.gz` (pre-computed) |

## 3. Build pQTL annotations

Count genome-wide-significant (P < 5e-8) cis and trans pQTLs per SNP across all proteins.

Output: `DATA/sumstats/{dataset}.pqtl.tsv` (per-SNP cis/trans/both counts)

## 4. Train SBayesRC weights

For each protein × cohort, train SBayesRC with functional annotations.

| Script | Output |
|--------|--------|
| `train_sbayesrc.py` | `DATA/sbayesrc/{dataset}.{ldm}.{annot}/{pid}.tsv.gz` |

Annotation models: `baseline+cis`, `baseline+cis+pqtl`. LD panels: HM3, Imputed. Runtime: ~720 min/protein via submitit.

```
DATA/
├── sumstats/
│   ├── {dataset}.gene.tsv
│   ├── {dataset}.pqtl.tsv
│   └── {dataset}/{pid}.ma.gz
└── sbayesrc/
    └── {dataset}.{ldm}.{annot}/{pid}.tsv.gz
```

---

# GWAS Compilation

Select independent traits and compile GWAS summary statistics.

## 5. Select independent traits

Use LDSC genetic correlations to pick ~independent traits (r² < 0.25), prioritizing high heritability.

| Script | Source | Output |
|--------|--------|--------|
| `ukb_indep_traits.py` | UKB BOLT-LMM | `DATA/trait_info.tsv`, `DATA/indep_traits.tsv`, `DATA/trait_values.tsv` |
| `pass_indep_traits.py` | PASS consortium | `DATA/pass_indep_traits.tsv` |

Result: 32 UKB PRICE2 traits + 56 PASS traits = 88 total.

## 6. Compile GWAS sumstats

Format to .ma aligned with LDM reference SNPs via SBayesRC munging.

| Script | Source | Output |
|--------|--------|--------|
| `compile_price2_sumstats.py` | UKB (excl PPP) | `DATA/PRICE2_GWAS/{trait}.ma` |
| `compile_pass_sumstats.py` | PASS consortium | `DATA/PASS_GWAS/{trait}.ma` |

```
DATA/
├── trait_info.tsv
├── indep_traits.tsv
├── pass_indep_traits.tsv
├── trait_values.tsv
├── PRICE2_GWAS/{trait}.ma
└── PASS_GWAS/{trait}.ma
```

---

# Protein Prediction

Compute polygenic scores from SBayesRC weights and evaluate prediction accuracy. Depends on: pqtl-compilation (weights), gwas-compilation (trait values).

## 7. Extract genotypes

Extract individual-level UKB genotypes for accuracy (~17K, with measured proteins) and imputation (~50K) cohorts.

| Script | Output |
|--------|--------|
| `extract_genotype.py` | `DATA/genotype/merged.pgen`, individual lists |

## 8. Compute polygenic scores

PLINK2 `--score` using SBayesRC weights, stratified by SNP subset (cis, trans, cisgenic, cisnongenic, cisexonic, cisnonexonic).

| Script | Output |
|--------|--------|
| `run_prediction.py` | `DATA/prediction/{group}.{subset}.parquet` |

## 9. Evaluate prediction accuracy

Correlate predicted vs measured protein levels in held-out UKB individuals.

| Script | Output |
|--------|--------|
| `eval_ukb_acc.py` | `DATA/predacc_stats.tsv` |

## 10. Variance explained in traits

Regress predicted protein PCs on 10 UKB traits, quantify cis/trans R².

| Script | Output |
|--------|--------|
| `run_variance_explained.py` | `DATA/variance-explained/{group}.{trait}.{n_pc}pc.json` |

```
DATA/
├── genotype/merged.{pgen,pvar,psam}
├── prediction/{group}.{subset}.parquet
├── predacc_stats.tsv
└── variance-explained/{group}.{trait}.{n_pc}pc.json
```

---

# PWAS Analysis

Compute PWAS Z-scores, co-regulation matrices, and validate against independent evidence. Depends on: pqtl-compilation (weights, annotations), gwas-compilation (GWAS .ma files).

## 11. Compute PWAS covariances

For each protein × trait, compute Z-score numerators (weight · GWAS-z) stratified by cis/trans.

| Script | Output |
|--------|--------|
| `compute_pwas.py` | `DATA/pwas/{gwas_group}/{pqtl_group}.tsv.gz` |

## 12. Compute co-regulation matrices

Protein × protein covariance (w'Rw) for cis and trans components.

| Script | Output |
|--------|--------|
| `compute_coreg.py` | `DATA/coreg/{group}.{cis\|trans}.parquet` |

## 13. Compute variance components

Diagonal of co-regulation (w'Rw per protein) for Z-score denominators.

| Script | Output |
|--------|--------|
| `compute_var.py` | `DATA/var/{group}.{cis\|trans}.parquet` |

## 14. Compile PWAS results

Combine covariance + variance → Z-scores, apply co-regulation PC regression (n_pc=20), bin by signal strength.

| Script | Output |
|--------|--------|
| `compile_pwas_df.py` | `DATA/pwas/{gwas_group}.{pqtl_group}.tsv` (CIS_Z, TRANS_Z columns) |

## 15. Optimize PC regression

Tune number of co-regulation PCs to remove using genomic jackknife. Tests n_pc ∈ {0, 5, 10, 15, 20}.

| Script | Output |
|--------|--------|
| `optimize_regress_pc.py` | `DATA/optimize_regress_pc/{group}.n_pc{n}.tsv` |

## 16. Burden test & PoPS validation

Test whether PWAS-significant proteins are enriched for rare-variant burden hits and PoPS pathway scores.

| Script | Output |
|--------|--------|
| `run_price_validation.py` | `DATA/price_validation/{group}.{tsv,enrichment.tsv,regression.tsv}` |

```
DATA/
├── pwas/{gwas_group}/{pqtl_group}.tsv.gz
├── pwas/{gwas_group}.{pqtl_group}.tsv
├── coreg/{group}.{cis|trans}.parquet
├── var/{group}.{cis|trans}.parquet
├── optimize_regress_pc/
└── price_validation/
```

---

# Simulation

FPR/power simulations for PWAS method validation. Depends on: pqtl-compilation (LDM directory). Runs independently of GWAS/PWAS stages.

## 17. Run simulations

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
