"""LD matrix operations using SBayesRC eigendecomposition format."""

import numpy as np
import pandas as pd
import os


class BlockLDM:
    """Block-diagonal LD matrix stored as eigendecompositions.

    Reads the SBayesRC binary eigendecomposition format:
    - snp.info: SNP metadata with Block assignments
    - ldm.info: block metadata
    - block{N}.eigen.bin: binary eigendecomposition per block

    Parameters
    ----------
    ldm_dir : str
        Path to the SBayesRC LD matrix directory.
    """

    def __init__(self, ldm_dir):
        self.ldm_dir = ldm_dir
        self.snp_info = pd.read_csv(f"{ldm_dir}/snp.info", sep="\t", index_col="ID")
        self.ldm_info = pd.read_csv(f"{ldm_dir}/ldm.info", sep="\t", index_col="Block")

    @property
    def n_snp(self):
        return len(self.snp_info)

    @property
    def n_block(self):
        return len(self.ldm_info)

    def read_block_eig(self, block: int):
        """Read eigendecomposition for a block from binary file.

        Returns
        -------
        dict with keys: m, k, lambda_sum, thresh, lambdas, U
        """
        ldfile = f"{self.ldm_dir}/block{block}.eigen.bin"
        if not os.path.exists(ldfile):
            raise FileNotFoundError(f"Cannot find LD file {ldfile}")

        with open(ldfile, "rb") as f:
            m = np.frombuffer(f.read(4), dtype=np.int32)[0]
            k = np.frombuffer(f.read(4), dtype=np.int32)[0]
            lambda_sum = np.frombuffer(f.read(4), dtype=np.float32)[0]
            thresh = np.frombuffer(f.read(4), dtype=np.float32)[0]
            lambdas = np.frombuffer(f.read(4 * k), dtype=np.float32)
            U = np.frombuffer(f.read(4 * m * k), dtype=np.float32).reshape((k, m)).T

        return {
            "m": m,
            "k": k,
            "lambda_sum": float(lambda_sum),
            "thresh": float(thresh),
            "lambdas": lambdas.astype(float),
            "U": U.astype(float),
        }

    def read_block_mat(self, block: int):
        """Reconstruct full LD matrix for a block: U @ diag(lambdas) @ U.T"""
        eig = self.read_block_eig(block)
        return eig["U"] @ np.diag(eig["lambdas"]) @ eig["U"].T

    def block_dot(self, idx: int, mat):
        """Compute LD @ mat for one block.

        Parameters
        ----------
        idx : int
            Block index.
        mat : DataFrame, Series, or ndarray
            Input matrix/vector aligned to block SNPs.
        """
        ldm = self.read_block_mat(idx)
        snps = self.snp_info[self.snp_info["Block"] == idx].index
        assert np.array_equal(snps, mat.index), f"SNPs do not match"

        if isinstance(mat, pd.DataFrame):
            return pd.DataFrame(ldm @ mat.values, index=mat.index, columns=mat.columns)
        elif isinstance(mat, pd.Series):
            return pd.Series(ldm @ mat.values, index=mat.index)
        elif isinstance(mat, np.ndarray):
            return ldm @ mat
        else:
            raise TypeError(f"Input must be DataFrame, Series, or ndarray, got {type(mat)}")

    def block_qf(self, idx: int, mat):
        """Compute quadratic form diag(mat.T @ LD @ mat) for one block."""
        snps = self.snp_info[self.snp_info["Block"] == idx].index
        assert np.array_equal(snps, mat.index), f"SNPs do not match"

        ldm = self.read_block_mat(idx)
        mat_values = mat.values
        qf = (mat_values * (ldm @ mat_values)).sum(axis=0)

        if isinstance(mat, pd.DataFrame):
            return pd.Series(qf, index=mat.columns)
        elif isinstance(mat, pd.Series):
            return qf.item()
        else:
            raise TypeError(f"Input must be DataFrame or Series, got {type(mat)}")

    def block_cov(self, idx: int, mat, eigen: bool = True):
        """Compute covariance matrix mat.T @ LD @ mat for one block."""
        is_ndarray = isinstance(mat, np.ndarray)
        if is_ndarray:
            mat_values = mat
        else:
            snps = self.snp_info[self.snp_info["Block"] == idx].index
            assert np.array_equal(snps, mat.index), f"SNPs do not match"
            mat_values = mat.values

        if eigen:
            eig = self.read_block_eig(idx)
            eigvecs, eigvals = eig["U"], eig["lambdas"]
            proj = eigvecs.T @ mat_values
            cov = (proj.T * eigvals) @ proj
        else:
            ldm = self.read_block_mat(idx)
            cov = mat_values.T @ ldm @ mat_values

        if is_ndarray:
            return cov
        elif isinstance(mat, pd.DataFrame):
            return pd.DataFrame(cov, index=mat.columns, columns=mat.columns)
        elif isinstance(mat, pd.Series):
            return cov.item()
        else:
            raise TypeError(f"Input must be DataFrame, Series, or ndarray, got {type(mat)}")

    def qf(self, mat: pd.DataFrame):
        """Compute quadratic form diag(mat.T @ LD @ mat) across all blocks."""
        blocks = self.snp_info.loc[mat.index, "Block"].unique()
        qf = 0.0
        for block_idx in blocks:
            block_snps = self.snp_info[self.snp_info["Block"] == block_idx].index
            block_mat = mat.reindex(block_snps).fillna(0.0)
            qf += self.block_qf(block_idx, block_mat)
        return qf

    def cov(self, mat: pd.DataFrame):
        """Compute covariance matrix mat.T @ LD @ mat across all blocks."""
        blocks = self.snp_info.loc[mat.index, "Block"].unique()
        cov = 0.0
        for block_idx in blocks:
            block_snps = self.snp_info[self.snp_info["Block"] == block_idx].index
            block_mat = mat.reindex(block_snps).fillna(0.0)
            cov += self.block_cov(block_idx, block_mat)
        return cov


def read_ldm(bin_path: str):
    """Read a full (non-block) LD matrix from binary file.

    Returns
    -------
    ldm : pd.DataFrame
        LD correlation matrix.
    ldm_info : pd.DataFrame
        SNP info for this LD matrix.
    """
    ldm_info = pd.read_csv(bin_path.replace(".bin", ".info"), sep=r"\s+")
    n = len(ldm_info)
    with open(bin_path, "rb") as f:
        ldm = pd.DataFrame(
            np.fromfile(f, dtype=np.float32, count=n * n).reshape(n, n),
            index=ldm_info["ID"],
            columns=ldm_info["ID"],
        )
    return ldm.astype(float), ldm_info
