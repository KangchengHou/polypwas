"""Block-aligned parquet storage for SNP × feature matrices."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


class BlockWgt:
    """Parquet-backed SNP × feature matrix with block-aligned row groups.

    Stores a (n_snp, n_features) float32 matrix where features are protein
    weights or GWAS Z-scores. Each row group corresponds to one LD block.

    File layout:
        columns: chrom (int8), pos (int32), feat_1 (float32), ..., feat_N (float32)
        row groups: one per LD block, ordered by genomic position
        metadata: {"feature_ids": [...]}

    Parameters
    ----------
    path : str
        Path to parquet file.
    """

    def __init__(self, path: str):
        self._path = path
        self._pf = pq.ParquetFile(path)
        self._feature_cols = [
            c for c in self._pf.schema_arrow.names if c not in ("chrom", "pos")
        ]

    # --- Construction ---

    @classmethod
    def from_weight_files(
        cls,
        paths: dict[str, str],
        snp_info: pd.DataFrame,
        out_path: str,
        value_col: str = "BETA",
    ) -> "BlockWgt":
        """Convert per-protein SBayesRC weight files to parquet.

        Reads weight files in streaming fashion (block by block) to avoid
        loading all 7M+ rows × 2800 proteins into memory at once.

        Parameters
        ----------
        paths : dict[str, str]
            {protein_id: path_to_weights.tsv.gz} mapping.
        snp_info : pd.DataFrame
            SNP info from BlockLDM with columns Chrom, PhysPos, Block.
        out_path : str
            Output parquet file path.
        value_col : str
            Column name to extract from weight files.
        """
        from .utils import MultiTableReader

        pids = list(paths.keys())
        reader = MultiTableReader(paths, value_col=value_col)

        schema = pa.schema(
            [("chrom", pa.int8()), ("pos", pa.int32())]
            + [(pid, pa.float32()) for pid in pids]
        )
        writer = pq.ParquetWriter(out_path, schema, compression="zstd")

        for block_idx, block_info in tqdm(
            snp_info.groupby("Block"), desc="Writing parquet blocks"
        ):
            nsnp = len(block_info)
            wgt_mat = reader.next(nsnp)
            block_data = {
                "chrom": block_info["Chrom"].values.astype(np.int8),
                "pos": block_info["PhysPos"].values.astype(np.int32),
            }
            for j, pid in enumerate(pids):
                block_data[pid] = wgt_mat.iloc[:, j].values.astype(np.float32)
            writer.write_table(pa.table(block_data, schema=schema))

        writer.close()
        return cls(out_path)

    @classmethod
    def from_gwas_files(
        cls,
        paths: dict[str, str],
        snp_info: pd.DataFrame,
        out_path: str,
    ) -> "BlockWgt":
        """Convert per-trait GWAS sumstats (.ma with b/se columns) to parquet Z-scores.

        Parameters
        ----------
        paths : dict[str, str]
            {trait: path_to_gwas.ma} mapping.
        snp_info : pd.DataFrame
            SNP info from BlockLDM with columns Chrom, PhysPos, Block.
        out_path : str
            Output parquet file path.
        """
        traits = list(paths.keys())
        snp_ids = snp_info.index

        all_z = np.empty((len(snp_info), len(traits)), dtype=np.float32)
        for j, (trait, path) in enumerate(
            tqdm(paths.items(), desc="Reading GWAS files")
        ):
            df = pd.read_csv(path, sep="\t", index_col="SNP")
            df = df.loc[snp_ids]
            all_z[:, j] = (df["b"] / df["se"]).values.astype(np.float32)

        schema = pa.schema(
            [("chrom", pa.int8()), ("pos", pa.int32())]
            + [(trait, pa.float32()) for trait in traits]
        )
        writer = pq.ParquetWriter(out_path, schema, compression="zstd")

        offset = 0
        for block_idx, block_info in tqdm(
            snp_info.groupby("Block"), desc="Writing parquet blocks"
        ):
            nsnp = len(block_info)
            block_data = {
                "chrom": block_info["Chrom"].values.astype(np.int8),
                "pos": block_info["PhysPos"].values.astype(np.int32),
            }
            for j, trait in enumerate(traits):
                block_data[trait] = all_z[offset : offset + nsnp, j]
            writer.write_table(pa.table(block_data, schema=schema))
            offset += nsnp

        writer.close()
        return cls(out_path)

    # --- Block-wise streaming ---

    @property
    def n_blocks(self) -> int:
        return self._pf.metadata.num_row_groups

    def read_block(self, i: int, columns: list[str] = None) -> np.ndarray:
        """Read one row group as (n_snp_in_block, n_features) float32 array."""
        cols = columns or self._feature_cols
        table = self._pf.read_row_group(i, columns=cols)
        return table.to_pandas().values.astype(np.float32)

    def read_block_pos(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Read (chrom, pos) for one row group."""
        table = self._pf.read_row_group(i, columns=["chrom", "pos"])
        df = table.to_pandas()
        return df["chrom"].values, df["pos"].values

    def iter_blocks(self, columns: list[str] = None):
        """Yield (block_index, values_array) for all row groups."""
        for i in range(self.n_blocks):
            yield i, self.read_block(i, columns)

    # --- Column access ---

    def read_column(self, col: str) -> np.ndarray:
        """Read one feature across all SNPs."""
        table = self._pf.read(columns=[col])
        return table.column(col).to_numpy().astype(np.float32)

    def read_columns(self, cols: list[str]) -> np.ndarray:
        """Read subset of features across all SNPs."""
        table = self._pf.read(columns=cols)
        return table.to_pandas().values.astype(np.float32)

    # --- Region queries ---

    def read_region(
        self, chrom: int, start: int, end: int, columns: list[str] = None,
    ) -> np.ndarray:
        """Read features for SNPs in a genomic region.

        Uses parquet row group statistics to skip irrelevant blocks.
        """
        cols = columns or self._feature_cols
        table = pq.read_table(
            self._path,
            columns=["chrom", "pos"] + cols,
            filters=[
                ("chrom", "=", chrom),
                ("pos", ">=", start),
                ("pos", "<=", end),
            ],
        )
        df = table.to_pandas()
        return df[cols].values.astype(np.float32)

    # --- Properties ---

    @property
    def columns(self) -> list[str]:
        return self._feature_cols

    @property
    def n_snp(self) -> int:
        return self._pf.metadata.num_rows

    @property
    def n_features(self) -> int:
        return len(self._feature_cols)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_snp, self.n_features)

    @property
    def block_sizes(self) -> list[int]:
        meta = self._pf.metadata
        return [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]

    def __repr__(self) -> str:
        return (
            f"BlockWgt({self._path!r}, "
            f"{self.n_snp:,} SNPs × {self.n_features} features, "
            f"{self.n_blocks} blocks)"
        )
