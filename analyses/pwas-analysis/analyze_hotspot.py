"""Analyze impact of pQTL hotspot filtering on PWAS results.

Compares default vs nohotspot (top 0.1% pQTL SNPs removed) across:
1. Per-trait trans variance explained drop
2. Per-trait significant association count drop
3. Per-trait GWAS hotspot enrichment among GWS SNPs
4. Binary enrichment (|cis|>10 & |trans|>10) for BURDEN/POPS
5. Logistic regression pseudo-R² (Z bins, Z bins + coreg bins)

Outputs to DATA/hotspot_analysis/:
- per_trait_summary.tsv
- binary_enrichment.tsv
- logistic_regression.tsv
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings
from pathlib import Path

from polypwas.stats import binom_ratio
from polypwas.utils import normalize_coreg

warnings.filterwarnings("ignore")

EXTERNAL = Path(__file__).parent.parent / "external"
DATA_DIR = Path(__file__).parent / "DATA"
Z_THRESH = 4.265  # |Z| > 4.265 ≈ p < 1e-5
BINS = [-1, 5, 10, 15, np.inf]
LABELS = ["<5", "5-10", "10-15", ">=15"]


def pseudo_rsq(model):
    """Nagelkerke pseudo-R²."""
    ll_null, ll_model, n = model.llnull, model.llf, model.nobs
    r2_coxsnell = 1 - np.exp((2 / n) * (ll_null - ll_model))
    return r2_coxsnell / (1 - np.exp((2 / n) * ll_null))


def compile_validation_data(dataset: str):
    """Compile burden + PoPS validation data for a dataset."""
    burden_df = (
        pd.read_csv(EXTERNAL / "gwas" / "burden.tsv", sep="\t")
        .assign(
            BURDEN_PVALUE=lambda x: 2 * stats.norm.sf(np.abs(x["beta"] / x["standard_error"])),
        )
        .rename(columns={"trait_id": "TRAIT", "ensg": "ENSEMBL"})
    )[["TRAIT", "ENSEMBL", "BURDEN_PVALUE"]]

    pops_df = (
        pd.read_csv(EXTERNAL / "gwas" / "pops.tsv", sep="\t")
        .rename(columns={"trait_id": "TRAIT", "ENSGID": "ENSEMBL", "PoPS_Score": "POPS_SCORE"})
    )[["TRAIT", "ENSEMBL", "POPS_SCORE"]]

    gene_df = pd.read_csv(
        EXTERNAL / "pqtl" / "sumstats" / f"{dataset}.gene.tsv", sep="\t",
    )[["ID", "ENSEMBL"]]

    gene_df = pd.merge(gene_df, burden_df, on="ENSEMBL")
    gene_df = pd.merge(gene_df, pops_df, on=["TRAIT", "ENSEMBL"])

    gene_df["BURDEN_SIGNIF"] = 0
    gene_df["POPS_TOP"] = 0
    for trait, trait_df in gene_df.groupby("TRAIT"):
        burden_signif = trait_df.BURDEN_PVALUE < 0.05 / len(trait_df)
        gene_df.loc[trait_df.index[burden_signif], "BURDEN_SIGNIF"] = 1
        gene_df.loc[
            trait_df.nlargest(burden_signif.sum(), "POPS_SCORE").index, "POPS_TOP"
        ] = 1

    return gene_df[["TRAIT", "ID", "BURDEN_SIGNIF", "POPS_TOP"]]


def compute_coreg_scores(group: str, subset: str):
    """Compute per-protein co-regulation score: sum of squared normalized coreg row."""
    C = pd.read_parquet(DATA_DIR / "coreg" / f"{group}.{subset}.parquet")
    valid = np.diag(C) > 0
    C = C.loc[valid, valid]
    C = normalize_coreg(C)
    return (C**2).sum(axis=1)


def compute_hotspot_enrichment(pqtl_count_path: str, gwas_dir: Path, traits: list):
    """Compute per-trait hotspot enrichment among GWS SNPs."""
    pqtl = pd.read_csv(pqtl_count_path, sep="\t", index_col=0, usecols=["SNP", "total_count"])
    threshold = pqtl["total_count"].quantile(0.999)
    hotspot_set = set(pqtl.index[pqtl["total_count"] > threshold])
    baseline_rate = len(hotspot_set) / len(pqtl)

    results = {}
    for trait in traits:
        ma_path = gwas_dir / f"{trait}.ma"
        if not ma_path.exists():
            results[trait] = np.nan
            continue
        ma = pd.read_csv(ma_path, sep="\t", usecols=["SNP", "b", "se"]).set_index("SNP")
        z = ma["b"] / ma["se"]
        p = 2 * stats.norm.sf(np.abs(z))
        sig = z.index[p < 5e-8].intersection(pqtl.index)
        if len(sig) < 10:
            results[trait] = 0.0
            continue
        n_hot = sum(1 for s in sig if s in hotspot_set)
        results[trait] = (n_hot / len(sig)) / baseline_rate
    return results


def per_trait_summary(pwas_group: str, ve_group: str, pqtl_dataset: str):
    """Compute per-trait summary: R² drop, #assoc drop, GWS enrichment."""
    # Significant associations
    df = pd.read_csv(DATA_DIR / "ppc_pwas" / f"{pwas_group}.tsv", sep="\t")
    for col in ["TRANS_Z", "TRANS_Z_NOHOTSPOT"]:
        df[col] = df[col].fillna(0)
    sub = df[df.N_PPC == 20]

    assoc_stats = {}
    for trait, tdf in sub.groupby("TRAIT"):
        n_def = (tdf.TRANS_Z.abs() > Z_THRESH).sum()
        n_nh = (tdf.TRANS_Z_NOHOTSPOT.abs() > Z_THRESH).sum()
        assoc_stats[trait] = (n_nh - n_def) / n_def * 100 if n_def > 0 else 0.0

    # Hotspot enrichment
    enrich = compute_hotspot_enrichment(
        EXTERNAL / "pqtl" / "sumstats" / f"{pqtl_dataset}.pqtl_count.tsv",
        EXTERNAL / "gwas" / "price2_compiled",
        list(assoc_stats.keys()),
    )

    # Variance explained
    ve_dir = Path(__file__).parent.parent / "indiv-level" / "DATA" / "var_explained"
    rows = []
    for f in sorted(ve_dir.glob(f"{ve_group}.*.tsv")):
        trait = f.stem.split(f"{ve_group}.")[-1]
        r = pd.read_csv(f, sep="\t")
        r20 = r[r["n_ppc"] == 20].iloc[0]
        ve_drop = (r20["trans_nohotspot_est"] - r20["trans_est"]) / abs(r20["trans_est"]) * 100
        rows.append({
            "trait": trait,
            "trans_rsq_drop_pct": ve_drop,
            "trans_assoc_drop_pct": assoc_stats.get(trait, np.nan),
            "gws_hotspot_enrichment": enrich.get(trait, np.nan),
        })

    return pd.DataFrame(rows).sort_values("trans_rsq_drop_pct")


def binary_enrichment(group: str):
    """Binary enrichment: |cis|>10 & |trans|>10 for default vs nohotspot."""
    dataset = group.split(".")[0]
    pwas_df = pd.read_csv(DATA_DIR / "ppc_pwas" / f"{group}.tsv", sep="\t")
    valid_df = compile_validation_data(dataset)
    merged = pd.merge(valid_df, pwas_df, on=["TRAIT", "ID"])
    for col in ["CIS_Z", "TRANS_Z", "CIS_Z_NOHOTSPOT", "TRANS_Z_NOHOTSPOT"]:
        merged[col] = merged[col].fillna(0)

    rows = []
    for n_ppc in [0, 10, 20, 30]:
        ppc_df = merged[merged.N_PPC == n_ppc]
        for label, cis_col, trans_col in [("default", "CIS_Z", "TRANS_Z"),
                                           ("nohotspot", "CIS_Z_NOHOTSPOT", "TRANS_Z_NOHOTSPOT")]:
            subset = ppc_df[(ppc_df[cis_col].abs() > 10) & (ppc_df[trans_col].abs() > 10)]
            for target in ["BURDEN_SIGNIF", "POPS_TOP"]:
                logrr, logrr_se = binom_ratio(
                    x1=int(subset[target].sum()), n1=len(subset),
                    x2=int(ppc_df[target].sum()), n2=len(ppc_df),
                )
                rows.append({
                    "group": group, "n_ppc": n_ppc, "variant": label,
                    "target_var": target,
                    "n_hit": int(subset[target].sum()), "n_subset": len(subset),
                    "logrr": logrr, "logrr_se": logrr_se,
                })
    return pd.DataFrame(rows)


def logistic_regression(group: str):
    """Logistic regression pseudo-R²: Z bins and Z bins + coreg bins."""
    dataset = group.split(".")[0]
    pwas_df = pd.read_csv(DATA_DIR / "ppc_pwas" / f"{group}.tsv", sep="\t")
    valid_df = compile_validation_data(dataset)
    merged = pd.merge(valid_df, pwas_df, on=["TRAIT", "ID"])
    df = merged[merged.N_PPC == 20].copy()
    for col in ["CIS_Z", "TRANS_Z", "CIS_Z_NOHOTSPOT", "TRANS_Z_NOHOTSPOT"]:
        df[col] = df[col].fillna(0)

    # Z-score level features
    df["ABS_CIS_LEVEL"] = pd.cut(df["CIS_Z"].abs(), bins=BINS, labels=LABELS)
    df["ABS_TRANS_LEVEL"] = pd.cut(df["TRANS_Z"].abs(), bins=BINS, labels=LABELS)
    df["ABS_CIS_LEVEL_NH"] = pd.cut(df["CIS_Z_NOHOTSPOT"].abs(), bins=BINS, labels=LABELS)
    df["ABS_TRANS_LEVEL_NH"] = pd.cut(df["TRANS_Z_NOHOTSPOT"].abs(), bins=BINS, labels=LABELS)

    # Coreg score features
    for subset, suffix in [("cis", ""), ("trans", ""),
                           ("cis_nohotspot", "_NH"), ("trans_nohotspot", "_NH")]:
        base = "CIS" if "cis" in subset and "trans" not in subset else "TRANS"
        scores = compute_coreg_scores(group, subset)
        score_df = scores.reset_index()
        score_df.columns = ["ID", f"{base}_COREG_SCORE{suffix}"]
        df = pd.merge(df, score_df, on="ID", how="left")

    for suffix in ["", "_NH"]:
        for base in ["CIS", "TRANS"]:
            col = f"{base}_COREG_SCORE{suffix}"
            level_col = f"{base}_COREG_LEVEL{suffix}"
            if df[col].notna().sum() >= 4:
                df[level_col] = pd.qcut(df[col], q=4, duplicates="drop")
            else:
                df[level_col] = pd.NA

    formulas = [
        ("Z_bins", "ABS_CIS_LEVEL + ABS_TRANS_LEVEL",
                   "ABS_CIS_LEVEL_NH + ABS_TRANS_LEVEL_NH"),
        ("Z_bins+coreg", "ABS_CIS_LEVEL + ABS_TRANS_LEVEL + CIS_COREG_LEVEL + TRANS_COREG_LEVEL",
                         "ABS_CIS_LEVEL_NH + ABS_TRANS_LEVEL_NH + CIS_COREG_LEVEL_NH + TRANS_COREG_LEVEL_NH"),
    ]

    rows = []
    for target in ["BURDEN_SIGNIF", "POPS_TOP"]:
        for label, def_f, nh_f in formulas:
            if "coreg" in label.lower():
                coreg_cols = ["CIS_COREG_LEVEL", "TRANS_COREG_LEVEL",
                              "CIS_COREG_LEVEL_NH", "TRANS_COREG_LEVEL_NH"]
                mdf = df.dropna(subset=coreg_cols + [target]).copy()
            else:
                mdf = df.dropna(subset=[target]).copy()
            model_def = smf.logit(f"{target} ~ {def_f}", data=mdf).fit(disp=False)
            model_nh = smf.logit(f"{target} ~ {nh_f}", data=mdf).fit(disp=False)
            r2_def = pseudo_rsq(model_def)
            r2_nh = pseudo_rsq(model_nh)
            rows.append({
                "group": group, "target_var": target, "features": label,
                "default_rsq": r2_def, "nohotspot_rsq": r2_nh,
                "diff": r2_nh - r2_def,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out_dir = DATA_DIR / "hotspot_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = [
        "ukbsun.imputed.baseline+cis+pqtl",
        "ukb_linreg_0pc.imputed.baseline+cis+pqtl",
        "ukb_linreg_20pc.imputed.baseline+cis+pqtl",
    ]

    # 1. Per-trait summary (uses ukbsun for PWAS, ukb_linreg_0pc for var_explained)
    print("Computing per-trait summary...")
    trait_df = per_trait_summary(
        pwas_group="ukbsun.imputed.baseline+cis+pqtl",
        ve_group="ukb_linreg_0pc.imputed.baseline+cis+pqtl",
        pqtl_dataset="ukb_linreg_0pc",
    )
    trait_df.to_csv(out_dir / "per_trait_summary.tsv", sep="\t", index=False, float_format="%.4f")
    print(f"  Wrote {out_dir / 'per_trait_summary.tsv'}")

    # 2. Binary enrichment
    print("Computing binary enrichment...")
    binary_dfs = []
    for group in groups:
        binary_dfs.append(binary_enrichment(group))
    binary_df = pd.concat(binary_dfs, ignore_index=True)
    binary_df.to_csv(out_dir / "binary_enrichment.tsv", sep="\t", index=False, float_format="%.6g")
    print(f"  Wrote {out_dir / 'binary_enrichment.tsv'}")

    # 3. Logistic regression
    print("Computing logistic regression...")
    logreg_dfs = []
    for group in groups:
        logreg_dfs.append(logistic_regression(group))
    logreg_df = pd.concat(logreg_dfs, ignore_index=True)
    logreg_df.to_csv(out_dir / "logistic_regression.tsv", sep="\t", index=False, float_format="%.6g")
    print(f"  Wrote {out_dir / 'logistic_regression.tsv'}")

    print("Done.")
