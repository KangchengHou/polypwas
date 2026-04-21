# polypwas

**Proteome-wide association studies with cis and trans pQTL weights**

`polypwas` is a refactor of the internal TRANS-PWAS workflow into a cleaner Python package plus an analysis pipeline. It is built for large-scale PWAS runs that combine:

- block LD resources from [SBayesRC](https://github.com/zhilizheng/SBayesRC)
- protein weights trained from cis and trans pQTL summary statistics
- GWAS summary statistics aligned to the same SNP reference

The repository has two layers:

- `src/polypwas/` contains reusable LD, weight-store, and PWAS computation code
- `analyses/` contains the end-to-end production pipeline used to compile inputs, run PWAS, validate signals, and build manuscript assets

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

This installs the minimal dependencies needed for the demo workflow (numpy, pandas, scipy, pyyaml, tqdm). For the full analysis pipeline (submitit, pyarrow, pgenlib, scikit-learn, statsmodels), install with:

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
  --gene-chr 1 \
  --gene-start 63063158 \
  --gene-end 63071830
```

### Option B: Quick demo (assoc only, no R/SBayesRC needed)

If you want to skip training entirely, download pre-trained weights from the GitHub release:

```bash
polypwas download-example --include-weights
polypwas assoc \
  --weights data/examples/angptl3.wgts.gz \
  --gwas data/examples/ldl.ma.gz \
  --ldm-dir data/ldm/ukbEUR_HM3 \
  --gene-chr 1 \
  --gene-start 63063158 \
  --gene-end 63071830
```

### Expected output

With the bundled ANGPTL3 + LDL example, `polypwas assoc` prints:

```text
CIS_Z=17.213785
TRANS_Z=-44.954875
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

### Single-gene coordinates (`--gene-chr`, `--gene-start`, `--gene-end`)

For a single gene, you can provide the cis-window anchor directly instead of a gene-info table. `polypwas assoc` accepts `--gene-chr`, `--gene-start`, and `--gene-end`.

```text
--gene-chr 1
--gene-start 63063158
--gene-end 63071830
```

### Gene info (`--gene-info`)

For multiple genes, provide a tab-separated gene-info table with `ID`, `CHROM`, `START`, and `END`. `polypwas assoc` uses the first row of that table.

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