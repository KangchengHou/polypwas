"""Utility functions: file I/O, PLINK scoring, data transformations."""

import numpy as np
import pandas as pd
import subprocess
import gzip
import inspect
from typing import Callable, Optional
from .config import get_config


class MultiTableReader:
    """Stream-read multiple tab-separated files in parallel, returning combined DataFrames.

    All files must have identical headers. Reads a fixed number of lines at a time
    via next(), useful for block-wise processing without loading everything into memory.

    Parameters
    ----------
    paths : dict[str, str]
        {id: file_path} mapping.
    value_col : str or list[str]
        Column(s) to extract from each file.
    index_col : str, optional
        Column to use as row index.
    value_func : callable, optional
        Function to combine multiple value columns. Required when value_col is a list.
    delimiter : str
        Column delimiter (default tab).
    """

    def __init__(
        self,
        paths: dict[str, str],
        value_col: str | list[str],
        index_col: Optional[str] = None,
        value_func: Callable[..., pd.Series] | None = None,
        delimiter: str = "\t",
    ):
        self.paths = paths
        self.index_col = index_col
        self.value_col = value_col
        self.value_func = value_func
        self.delimiter = delimiter

        if isinstance(value_col, list) and value_func is None:
            raise ValueError("value_func must be provided when value_col is a list")

        self.handles = [
            gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")
            for path in paths.values()
        ]
        headers = [next(fh).strip().split("\t") for fh in self.handles]

        if not all(header == headers[0] for header in headers):
            raise ValueError("All input files must have identical headers")
        self.header = headers[0]

        required_cols = [value_col] if isinstance(value_col, str) else value_col
        for col in required_cols:
            if col not in self.header:
                raise ValueError(f"Files must contain column {col}")
        if (self.index_col is not None) and (self.index_col not in self.header):
            raise ValueError(f"Files must contain index column {self.index_col}")

        if (
            isinstance(value_col, list)
            and (value_func is not None)
            and (len(inspect.signature(value_func).parameters) != len(value_col))
        ):
            raise ValueError(
                f"value_func expects {len(inspect.signature(value_func).parameters)} "
                f"arguments but {len(value_col)} columns provided"
            )

    def next(self, nlines: int) -> pd.DataFrame:
        """Read next nlines from each file and return combined DataFrame."""
        chunks = []
        for handle in self.handles:
            df = pd.DataFrame(
                [next(handle).strip().split(self.delimiter) for _ in range(nlines)],
                columns=self.header,
            )
            if self.index_col is not None:
                df = df.set_index(self.index_col)

            if isinstance(self.value_col, str):
                chunk = df[self.value_col]
            else:
                cols = [df[col].astype(float) for col in self.value_col]
                chunk = self.value_func(*cols)

            chunks.append(chunk)

        return pd.DataFrame(
            {pid: chunk.astype(float) for pid, chunk in zip(self.paths.keys(), chunks)}
        )

    def __del__(self):
        for handle in self.handles:
            handle.close()


def score_plink(pfile, score_path, out_prefix, indiv_subset=None, snp_subset=None, memory=16000):
    """Run PLINK2 --score for genetic prediction.

    Parameters
    ----------
    pfile : str
        Path to PLINK2 pfile (without extension).
    score_path : str
        Path to score file (SNP, A1, BETA columns).
    out_prefix : str
        Output prefix for .sscore file.
    indiv_subset : str, optional
        Path to individual keep file.
    snp_subset : str, optional
        Path to SNP extract file.
    memory : int
        Memory limit in MB (default 16000).
    """
    plink2_path = get_config("plink2")
    cmds = [
        plink2_path,
        f"--pfile {pfile}",
        f"--score {score_path} 1 2 3 header-read "
        f"cols=+scoresums,-scoreavgs,-fid,-dosagesum,-nallele",
        f"--memory {memory} --silent",
        f"--out {out_prefix}",
    ]
    if indiv_subset is not None:
        cmds.append(f"--keep {indiv_subset}")
    if snp_subset is not None:
        cmds.append(f"--extract {snp_subset}")
    subprocess.run(" ".join(cmds), shell=True)


def inverse_rank_normalize(val: np.ndarray):
    """Inverse rank normalization to standard normal distribution.

    Handles NaN values by preserving their positions.
    """
    from scipy.stats import rankdata, norm

    val = np.array(val)
    non_nan_index = ~np.isnan(val)
    results = np.full(val.shape, np.nan)
    results[non_nan_index] = norm.ppf(
        (rankdata(val[non_nan_index]) - 0.5) / len(val[non_nan_index])
    )
    return results


def read_coreg(path: str, normalize: bool = True) -> pd.DataFrame:
    """Read co-regulation matrix from TSV, optionally normalizing to correlation scale."""
    coreg = pd.read_csv(path, sep="\t", index_col=0)
    if normalize:
        coreg = normalize_coreg(coreg)
    return coreg


def normalize_coreg(coreg):
    """Normalize covariance matrix to correlation matrix."""
    coreg = coreg.copy()
    diag = np.diag(coreg)
    coreg /= np.sqrt(diag[:, None])
    coreg /= np.sqrt(diag[None, :])
    return coreg


def get_exon_mask(exon_str, positions):
    """Return boolean mask for positions falling within any exon interval.

    Parameters
    ----------
    exon_str : str
        Comma-separated intervals, e.g. '100-200,300-400'.
    positions : array-like
        Genomic positions to test.
    """
    intervals = np.array([list(map(int, x.split("-"))) for x in exon_str.split(",")])
    starts = intervals[:, 0]
    ends = intervals[:, 1]
    positions = np.asarray(positions)
    return np.any((positions[:, None] >= starts) & (positions[:, None] <= ends), axis=1)
