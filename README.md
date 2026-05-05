# polypwas

Code base for [Functionally informed cis and trans proteome-wide association studies prioritize disease-critical genes](https://www.medrxiv.org/content/10.64898/2026.04.24.26351667v1).

> **Note:** This repository is still under active development.


## Prerequisites

### R and SBayesRC

The `polypwas train` step calls [SBayesRC](https://github.com/zhilizheng/SBayesRC) via `Rscript`. You need a working R installation with the SBayesRC package **before** running `polypwas setup`.

Install SBayesRC in R:

```r
install.packages(c("Rcpp", "data.table", "stringi", "BH", "RcppEigen"))
install.packages(
  "https://github.com/zhilizheng/SBayesRC/releases/download/v0.2.6/SBayesRC_0.2.6.tar.gz",
  repos = NULL, type = "source"
)
```

If you only want to run `polypwas assoc` with pre-trained weights (see below), SBayesRC is **not** required.

## Installation and example usage

Requires Python `>=3.9` and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:kangchenghou/polypwas.git
cd polypwas
uv sync
```

This installs the dependencies needed for the demo workflow and `polypwas assoc` (numpy, pandas, scipy, pyyaml, tqdm, pyarrow). For the full analysis pipeline (submitit, pgenlib, scikit-learn, statsmodels), install with:

```bash
uv sync --extra full
```

Then configure external tool paths and validate that `Rscript` can load SBayesRC:

```bash
polypwas setup                              # interactive prompt
polypwas setup --rscript /path/to/Rscript   # non-interactive
```

### Option A: Full demo (train + assoc)

This downloads example pQTL/GWAS summary statistics and the HM3 LD reference (~3 GB), trains SBayesRC weights (~3 min), and computes PWAS Z-scores.

```bash
polypwas download-example
polypwas train \
  --pqtl data/examples/angptl3.ma.gz \
  --ldm-dir data/ldm/ukbEUR_HM3 \
  --threads 10 \
  --out angptl3.wgts.gz

polypwas assoc \
  --weights angptl3.wgts.gz \
  --gwas data/examples/ldl.ma.gz \
  --ldm-dir data/ldm/ukbEUR_HM3 \
  --gene-info data/examples/angptl3.gene.tsv
```

### Option B: Quick demo (assoc only, no R/SBayesRC needed)

If you want to skip training entirely, download pre-trained weights from the GitHub release:

```bash
polypwas download-example --include-weights
polypwas assoc \
  --weights data/examples/angptl3.wgts.gz \
  --gwas data/examples/ldl.ma.gz \
  --ldm-dir data/ldm/ukbEUR_HM3 \
  --gene-info data/examples/angptl3.gene.tsv
```

### Option C: Batch (many proteins, single trait)

`polypwas assoc` also accepts a single `.parquet` `BlockWgt` file (n_snp × n_protein, block-aligned) plus a multi-row `--gene-info` and writes a TSV with `ID, CIS_Z, TRANS_Z`:

```bash
polypwas assoc --verbose \
  --weights   ukbsun.imputed.baseline+cis+pqtl.parquet \
  --gwas      biochemistry_LDLdirect.ma \
  --ldm-dir   ukbEUR_Imputed \
  --gene-info ukbsun.gene.tsv \
  --out       pwas_z.tsv
```

See `analyses/README.md` for the in-cluster dataset layout and a runnable example.

### Expected output

With the bundled ANGPTL3 + LDL example, `polypwas assoc` prints:

```text
CIS_Z=17.209574
TRANS_Z=-45.684548
```

(Exact values may vary slightly depending on SBayesRC random seed when training your own weights.)

### LD reference download

`polypwas download-example` automatically downloads the HapMap3 LD reference from the [SBayesRC resource page](https://github.com/zhilizheng/SBayesRC#resources). The primary download URL is:

```
https://gctbhub.cloud.edu.au/data/SBayesRC/resources/v2.0/LD/HapMap3/ukbEUR_HM3.zip
```

If automatic download fails, you can manually download from the [SBayesRC Google Drive mirror](https://drive.google.com/drive/folders/1uxnxDjRJPzo0dTpFnERS5N2NGZX5S-sU) and extract into `data/ldm/`.

> **Maintainer note:** The public example bundle is published as the GitHub release `angptl3-ldl-example`. To refresh those downloadable assets, run:
>
> ```bash
> gh release upload angptl3-ldl-example \
>   data/examples/angptl3.ma.gz data/examples/ldl.ma.gz angptl3.wgts.gz --clobber
> ```

## File formats

### pQTL summary statistics (`--pqtl`)

`data/examples/angptl3.ma.gz` is a gzipped tab-separated summary-statistics table with required columns `SNP`, `freq`, `b`, `se`, and `N`. If `A1`, `A2`, or `p` are missing, `polypwas train` fills them from `--ldm-dir/snp.info` and from `b / se`.

```text
SNP         freq        b          se         N
rs4970383   0.245983   -0.00552949 0.00803738 33671
rs4475691   0.198152   -0.0119935  0.00867688 33671
...
```

### LD reference (`--ldm-dir`)

`data/ldm/ukbEUR_HM3/` is a directory, not a single file. It must contain `snp.info`, `ldm.info`, and `block*.eigen.bin`. These are the SBayesRC eigendecomposition LD matrices; see [SBayesRC Resources](https://github.com/zhilizheng/SBayesRC#resources) for other ancestry/density options.

```text
ukbEUR_HM3/
├── snp.info
├── ldm.info
├── block1.eigen.bin
...
```

### Trained weights output (`--out`)

`angptl3.wgts.gz` is the final trained weights file written by `polypwas train`. SBayesRC sidecar formatting happens internally before the CLI writes this output.

```text
angptl3.wgts.gz
```

### GWAS summary statistics (`--gwas`)

`data/examples/ldl.ma.gz` is a gzipped tab-separated GWAS file indexed by `SNP`, with `b` and `se` used to form GWAS Z-scores.

```text
SNP         A1 A2 freq       b           se         p          N
rs4970383   A  C  0.247625  -0.006412   0.003861   0.09658    442817
rs4475691   T  C  0.199300  -0.001188   0.004159   0.7752     442817
...
```

### Gene info (`--gene-info`)

A tab-separated table with `ID`, `CHROM`, `START`, `END`. For a single weight file, the first row is used; for batch mode (parquet weights or many `.tsv.gz` files), every row is matched against the protein IDs.

```text
ID       CHROM     START       END
ANGPTL3  1         63063158    63071830
SORT1    1         109274968   109284742
...
```

### Trained weights file

The trained SBayesRC weights file used by `polypwas assoc` is expected to look like:

```text
SNP    A1    BETA    SE    PIP    BETAlast
rs4970383 A -0.000004 0.000351 0.0325 0
rs4475691 T -0.000013 0.000375 0.0270 0
...
```

## Notes
This repository was developed with assistance from AI coding software, including Claude Code and Codex.