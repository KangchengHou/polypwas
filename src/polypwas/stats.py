"""Statistical utilities for PWAS analysis."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union


# --- Correlation and partial correlation ---


def pcorr(X, Y=None, covar=None):
    """Correlation or partial correlation between variables.

    Parameters
    ----------
    X : ndarray or DataFrame, shape (n, p)
        First set of variables.
    Y : ndarray or DataFrame, shape (n, q), optional
        Second set. If None, computes correlation of X with itself.
    covar : ndarray, shape (n, k), optional
        Covariates to regress out before computing correlation.

    Returns
    -------
    ndarray or DataFrame
        Correlation matrix (p x q) or (p x p) if Y is None.
    """
    from scipy import linalg

    if Y is None:
        columns = X.columns if isinstance(X, pd.DataFrame) else None
        X = np.array(X)
        assert X.ndim == 2
        n = X.shape[0]

        if covar is not None:
            covar = np.array(covar)
            coef = linalg.lstsq(covar, X)[0]
            X = X - covar @ coef

        xv = X - X.mean(axis=0)
        xvss = (xv * xv).sum(axis=0)
        r = np.matmul(xv.T, xv)
        r /= np.sqrt(xvss)[:, None]
        r /= np.sqrt(xvss)
        r = np.clip(r, -1, 1)

        if columns is not None:
            r = pd.DataFrame(r, index=columns, columns=columns)
    else:
        if isinstance(X, pd.DataFrame):
            assert isinstance(Y, pd.DataFrame)
            columns1, columns2 = X.columns, Y.columns
        else:
            columns1, columns2 = None, None

        X, Y = np.array(X), np.array(Y)
        assert X.ndim == 2 and Y.ndim == 2
        assert X.shape[0] == Y.shape[0]
        n = X.shape[0]

        if covar is not None:
            covar = np.array(covar)
            coef = linalg.lstsq(covar, X)[0]
            X = X - covar @ coef
            coef = linalg.lstsq(covar, Y)[0]
            Y = Y - covar @ coef

        xv = X - X.mean(axis=0)
        yv = Y - Y.mean(axis=0)
        xvss = (xv * xv).sum(axis=0)
        yvss = (yv * yv).sum(axis=0)
        r = np.matmul(xv.T, yv)
        r /= np.sqrt(xvss)[:, None]
        r /= np.sqrt(yvss)
        r = np.clip(r, -1, 1)

        if columns1 is not None:
            r = pd.DataFrame(r, index=columns1, columns=columns2)

    return r


# --- R-squared ---


def rsquared(y: np.ndarray, X: np.ndarray):
    """Compute R-squared from individual-level data."""
    from scipy.linalg import lstsq

    resid = lstsq(a=X, b=y)[1]
    return 1 - resid / (y**2).sum()


def adjusted_rsquared(y: np.ndarray, X: np.ndarray):
    """Compute bias-corrected R-squared with standard error.

    Uses Cholesky decomposition for speed. Falls back to pinvh for
    rank-deficient X.T @ X.
    """
    import scipy.linalg

    n = len(y)
    assert X.shape[0] == n

    beta = X.T @ y / n
    V = X.T @ X / n

    try:
        L = scipy.linalg.cho_factor(V)
        quad_form = np.dot(beta, scipy.linalg.cho_solve(L, beta))
        rank = V.shape[0]
    except scipy.linalg.LinAlgError:
        inv_V, rank = scipy.linalg.pinvh(V, return_rank=True)
        quad_form = np.dot(beta, np.dot(inv_V, beta))

    est = (n * quad_form - rank) / (n - rank)
    var = ((n / (n - rank)) ** 2) * (2 * rank * (1 - est) / n + 4 * est) * (1 - est) / n
    return est, np.sqrt(var)


# --- Summary-statistics based methods ---


def sumstats_rsquared(XtX: np.ndarray, Xty: np.ndarray, idx: list[int]) -> float:
    """Compute R-squared from summary statistics.

    Assumes XtX, Xty are divided by N and y is standardized (yTy/N = 1).
    """
    idx = np.asarray(idx)
    XtX_sub = XtX[np.ix_(idx, idx)]
    Xty_sub = Xty[idx]
    beta = np.linalg.pinv(XtX_sub) @ Xty_sub
    return float(Xty_sub.T @ beta)


def sumstats_adjusted_rsquared(XtX: np.ndarray, Xty: np.ndarray, idx: list[int], n: int):
    """Compute bias-corrected R-squared from summary statistics."""
    from scipy.linalg import pinvh

    idx = np.asarray(idx)
    V = XtX[np.ix_(idx, idx)]
    beta = Xty[idx]

    inv_V, rank = pinvh(V, return_rank=True)
    quad_form = beta @ inv_V @ beta

    est = (n * quad_form - rank) / (n - rank)
    var = ((n / (n - rank)) ** 2) * (2 * rank * (1 - est) / n + 4 * est) * (1 - est) / n
    return est, np.sqrt(var)


def sumstats_pcorr(XtX: np.ndarray, Xty: np.ndarray, idx: list[int], cond_idx: list[int]):
    """Compute partial correlations from summary statistics.

    Assumes XtX, Xty divided by N, yTy/N = 1, diag(XtX) = 1.
    """
    idx = np.asarray(idx)
    cond_idx = np.asarray(cond_idx)

    if len(cond_idx) == 0:
        return Xty[idx]

    XtX_cond = XtX[np.ix_(cond_idx, cond_idx)]
    XtX_cross = XtX[np.ix_(cond_idx, idx)]
    Xty_cond = Xty[cond_idx]
    inv_XtX_cond = np.linalg.pinv(XtX_cond)

    proj_Xty = XtX_cross.T @ inv_XtX_cond @ Xty_cond
    proj_XtX = np.einsum("ij,jk,ki->i", XtX_cross.T, inv_XtX_cond, XtX_cross)

    numer = Xty[idx] - proj_Xty
    denom_y = np.sqrt(1 - Xty_cond.T @ inv_XtX_cond @ Xty_cond)
    denom_X = np.sqrt(np.maximum(1 - proj_XtX, 0))
    return numer / (denom_y * denom_X)


# --- Stepwise regression ---


def stepwise_regression(
    y: pd.Series,
    X: pd.DataFrame,
    pvalue_tol: float = 1e-5,
    min_n: int = 20,
    start_cols: list[str] = None,
    clump: bool = False,
    clump_r2: float = 0.1,
    verbose: bool = True,
):
    """Forward stepwise regression selecting predictors by partial correlation.

    Intercept is added automatically.
    """
    import statsmodels.api as sm

    n_indiv = len(y)
    rsq_df = {"selected": [], "rsquare": [], "pvalue": []}

    included_cols = []
    if start_cols is not None:
        for col in start_cols:
            score = (
                pcorr(
                    X=X[col].values[:, None],
                    Y=y.values[:, None],
                    covar=sm.add_constant(X[included_cols]).values,
                )
                ** 2
            ).item()
            pvalue = stats.chi2.sf(score * (n_indiv - 1), 1)
            rsq_df["selected"].append(col)
            included_cols.append(col)
            rsq = rsquared(y, sm.add_constant(X[included_cols]))
            incre_rsq = rsq - np.sum(rsq_df["rsquare"])
            rsq_df["rsquare"].append(incre_rsq)
            rsq_df["pvalue"].append(pvalue)
            if verbose:
                print(
                    f"{col} added: incremental R2 = {incre_rsq * 100:.2g}%, "
                    f"total R2 = {rsq * 100:.3g}%, P = {pvalue:.2g}"
                )

    remained_cols = [col for col in X.columns if col not in included_cols]

    while len(remained_cols) > 0:
        scores = (
            pcorr(
                X=X[remained_cols].values,
                Y=y.values[:, None],
                covar=sm.add_constant(X[included_cols]).values,
            ).flatten()
            ** 2
        )
        max_col = remained_cols[np.argmax(scores)]
        pvalue = stats.chi2.sf(scores[np.argmax(scores)] * (n_indiv - 1), 1)
        included_cols.append(max_col)
        if clump:
            r2_with_max = X[remained_cols].corrwith(X[max_col]) ** 2
            remained_cols = [col for col in remained_cols if r2_with_max[col] < clump_r2]
        else:
            remained_cols = [col for col in remained_cols if col != max_col]
        rsq = rsquared(y=y, X=sm.add_constant(X[included_cols]))
        incre_rsq = rsq - np.sum(rsq_df["rsquare"])
        rsq_df["selected"].append(max_col)
        rsq_df["rsquare"].append(incre_rsq)
        rsq_df["pvalue"].append(pvalue)
        if verbose:
            print(
                f"{max_col} added: incremental R2 = {incre_rsq * 100:.2g}%, "
                f"total R2 = {rsq * 100:.3g}%, P = {pvalue:.2g}"
            )
        if (len(included_cols) > min_n) and (pvalue > pvalue_tol):
            break

    return pd.DataFrame(rsq_df).set_index("selected")


def sumstats_stepwise_regression(
    XtX: pd.DataFrame,
    Xty: pd.Series,
    n_indiv: int,
    verbose: bool = True,
    pvalue_tol: float = 1e-5,
    min_n: int = 20,
):
    """Forward stepwise regression using summary statistics only."""
    included_cols = []
    remained_cols = list(XtX.columns)
    rsq_df = {"selected": [], "rsquare": [], "zscore": [], "pvalue": []}

    col_idx_dict = {col: i for i, col in enumerate(XtX.columns)}
    while len(remained_cols) > 0:
        pcorr_vals = sumstats_pcorr(
            XtX=XtX.values,
            Xty=Xty.values,
            idx=[col_idx_dict[col] for col in remained_cols],
            cond_idx=[col_idx_dict[col] for col in included_cols],
        )
        squared_pcorr = pcorr_vals**2

        max_col = remained_cols[np.argmax(squared_pcorr)]
        zscore = pcorr_vals[np.argmax(squared_pcorr)] * np.sqrt(n_indiv - 1)
        pvalue = stats.chi2.sf(zscore**2, 1)
        included_cols.append(max_col)
        remained_cols = [col for col in remained_cols if col != max_col]
        rsq = sumstats_rsquared(
            XtX=XtX.values,
            Xty=Xty.values,
            idx=[col_idx_dict[col] for col in included_cols],
        )
        adjusted_rsq, adjusted_rsq_se = sumstats_adjusted_rsquared(
            XtX=XtX.values,
            Xty=Xty.values,
            idx=[col_idx_dict[col] for col in included_cols],
            n=n_indiv,
        )
        incre_rsq = rsq - np.sum(rsq_df["rsquare"])
        rsq_df["selected"].append(max_col)
        rsq_df["rsquare"].append(incre_rsq)
        rsq_df["zscore"].append(zscore)
        rsq_df["pvalue"].append(pvalue)
        if verbose:
            print(
                f"({len(included_cols)}) {max_col} added: incremental R2 = {incre_rsq * 100:.2g}%, "
                f"total R2 = {rsq * 100:.3g}%, adjusted R2 = {adjusted_rsq * 100:.3g}%, "
                f"Z = {zscore:.2g}, P = {pvalue:.2g}"
            )
        if (len(included_cols) > min_n) and (pvalue > pvalue_tol):
            break

    return pd.DataFrame(rsq_df).set_index("selected")


# --- Co-regulation PC regression ---


def sumstats_regress_pc(XtX, Xty, n_pc=10, normalize_regressed=True):
    """Remove top n_pc principal components from summary statistics.

    Parameters
    ----------
    XtX : ndarray or DataFrame, shape (P, P)
        Sample covariance X.T @ X / N.
    Xty : ndarray or Series, shape (P,)
        Cross-covariance X.T @ y / N.
    n_pc : int
        Number of PCs to remove.
    normalize_regressed : bool
        Re-normalize after projection.

    Returns
    -------
    XtX_proj, Xty_proj
        Projected summary statistics (same type as input).
    """
    XtX_is_df = isinstance(XtX, pd.DataFrame)
    Xty_is_series = isinstance(Xty, (pd.Series, pd.DataFrame))

    XtX_values = XtX.values if XtX_is_df else XtX
    Xty_values = Xty.values if Xty_is_series else Xty

    eigvals, eigvecs = np.linalg.eigh(XtX_values)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    V_k = eigvecs[:, :n_pc]
    P = np.eye(XtX_values.shape[0]) - V_k @ V_k.T

    XtX_proj = P @ XtX_values @ P
    Xty_proj = P @ Xty_values

    if normalize_regressed:
        diag = np.diag(XtX_proj)
        XtX_proj = XtX_proj / np.sqrt(diag)[:, None] / np.sqrt(diag)[None, :]
        Xty_proj = Xty_proj / np.sqrt(diag)

    if XtX_is_df:
        XtX_proj = pd.DataFrame(XtX_proj, index=XtX.index, columns=XtX.index)
    if Xty_is_series:
        Xty_proj = pd.Series(Xty_proj, index=Xty.index)

    return XtX_proj, Xty_proj


def compute_coreg_pc(coreg: np.ndarray, n_pc: int):
    """Compute top eigenvectors/values of a normalized co-regulation matrix."""
    assert np.allclose(np.diag(coreg), 1, atol=1e-6), "coreg must be normalized"
    eigvals, eigvecs = np.linalg.eigh(coreg)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx][:n_pc], eigvecs[:, idx][:, :n_pc]


def regress_coreg_pc(eigvals, eigvecs, coreg=None, z=None, eps=1e-12):
    """Regress out co-regulation PCs from a correlation matrix or z-score vector.

    Exactly one of coreg or z must be provided.
    """
    assert (coreg is not None) + (z is not None) == 1

    if eigvals.size == 0 or eigvecs.size == 0:
        return coreg if coreg is not None else z

    P, k = eigvecs.shape
    assert eigvals.shape == (k,)
    diag_proj = 1.0 - (eigvecs**2) @ eigvals
    diag_proj = np.maximum(diag_proj, eps)
    s = 1.0 / np.sqrt(diag_proj)

    if coreg is not None:
        assert np.allclose(np.diag(coreg), 1, atol=1e-6)
        C_proj = coreg - eigvecs @ (eigvals[:, np.newaxis] * eigvecs.T)
        return (C_proj * s[:, None]) * s[None, :]

    assert z.shape == (P,)
    z_proj = z - eigvecs @ (eigvecs.T @ z)
    return z_proj * s


# --- Enrichment and effect size statistics ---


def rope_prob(
    slope: Union[float, np.ndarray],
    se: Union[float, np.ndarray],
    lower: float = -0.1,
    upper: float = 0.1,
):
    """Probability that a slope falls within a region of practical equivalence [lower, upper]."""
    return stats.norm.cdf((upper - slope) / se) - stats.norm.cdf((lower - slope) / se)


def compute_excess_overlap(n_overlap, n1, n2, n_total):
    """Compute enrichment (log odds ratio) for overlap between two sets.

    Returns (log_oddsratio, log_oddsratio_se, pvalue).
    """
    from scipy.stats import fisher_exact

    n_other = n_total - n1 - n2 + n_overlap
    oddsratio, pvalue = fisher_exact(
        [[n_other, n1 - n_overlap], [n2 - n_overlap, n_overlap]]
    )
    log_oddsratio = np.log(oddsratio)

    if any(x == 0 for x in [n1 - n_overlap, n2 - n_overlap, n_overlap, n_other]):
        log_oddsratio_se = 0
    else:
        log_oddsratio_se = np.sqrt(
            1 / (n1 - n_overlap) + 1 / (n2 - n_overlap) + 1 / n_overlap + 1 / n_other
        )

    return log_oddsratio, log_oddsratio_se, pvalue


def binom_ratio(x1: int, n1: int, x2: int, n2: int):
    """Log relative risk and SE for two binomial proportions."""
    a, b, c, d = x1, n1 - x1, x2, n2 - x2
    rr = (x1 / n1) / (x2 / n2)
    se_logrr = np.sqrt(1 / a + 1 / c - 1 / (a + b) - 1 / (c + d))
    return np.log(rr), se_logrr


def compute_ranked_oddsratio(ranked: list, target: list, totaln: int):
    """Compute enrichment odds ratios at each rank cutoff.

    Returns DataFrame indexed by n with columns: log_oddsratio, log_oddsratio_se, pvalue.
    """
    from scipy.stats import fisher_exact

    target = set(target)
    results = []

    for n in range(1, len(ranked) + 1):
        top_n = set(ranked[:n])
        n_overlap = len(top_n & target)
        n_other = totaln - len(target) - (n - n_overlap)

        oddsratio, pvalue = fisher_exact(
            [[n_other, n - n_overlap], [len(target) - n_overlap, n_overlap]]
        )
        log_oddsratio = np.log(oddsratio + 1e-100)

        if any(x == 0 for x in [n - n_overlap, len(target) - n_overlap, n_overlap, n_other]):
            log_oddsratio_se = 0
        else:
            log_oddsratio_se = np.sqrt(
                1 / (n - n_overlap) + 1 / (len(target) - n_overlap) + 1 / n_overlap + 1 / n_other
            )

        results.append({
            "n": n,
            "log_oddsratio": log_oddsratio,
            "log_oddsratio_se": log_oddsratio_se,
            "pvalue": pvalue,
        })

    return pd.DataFrame(results).set_index("n")


def sumstats_mr(beta_x, se_x, beta_y, se_y):
    """Single-variant Mendelian Randomization (Wald ratio).

    Returns dict with keys: ratio, se, z, p.
    """
    beta_x, beta_y = float(beta_x), float(beta_y)
    se_x, se_y = float(se_x), float(se_y)

    if beta_x == 0.0:
        raise ZeroDivisionError("beta_x is 0; Wald ratio is undefined.")

    theta = beta_y / beta_x
    var_theta = (se_y**2) / (beta_x**2) + (beta_y**2) * (se_x**2) / (beta_x**4)
    se_theta = np.sqrt(var_theta)
    z = theta / se_theta
    p = stats.norm.sf(np.abs(z)) * 2

    return {"ratio": float(theta), "se": float(se_theta), "z": float(z), "p": float(p)}


def pseudo_rsq(model):
    """Nagelkerke pseudo R-squared from a fitted statsmodels model."""
    ll_null = model.llnull
    ll_model = model.llf
    n = model.nobs
    r2_coxsnell = 1 - np.exp((2 / n) * (ll_null - ll_model))
    r2_nagelkerke = r2_coxsnell / (1 - np.exp((2 / n) * ll_null))
    return r2_nagelkerke


def bootstrap_median_stderr(values, n_bootstrap=1000):
    """Bootstrap standard error of the median."""
    rng = np.random.default_rng(42)
    bootstrap_medians = [
        np.median(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ]
    return np.std(bootstrap_medians)
