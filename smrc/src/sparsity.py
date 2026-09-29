"""Measuring how much memory sparse storage saves.

A CSR matrix keeps only its non-zero values plus two index arrays, so its real
footprint is the size of those three arrays. This module compares that against
the size a dense array of the same shape would need.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from scipy.sparse import csr_matrix, issparse


@dataclass
class SparsityReport:
    n_rows: int
    n_cols: int
    nnz: int
    density: float
    sparsity: float
    sparse_mb: float
    dense_mb: float
    savings_ratio: float

    def as_row(self) -> dict:
        return asdict(self)


def analyse(X, bytes_per_value: int = 8) -> SparsityReport:
    """Compare the CSR footprint of X against a dense float64 equivalent."""
    X = csr_matrix(X) if not issparse(X) else X.tocsr()
    n_rows, n_cols = X.shape
    total = n_rows * n_cols
    density = X.nnz / total if total else 0.0

    sparse_bytes = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    dense_bytes = total * bytes_per_value
    mb = 1024 ** 2

    return SparsityReport(
        n_rows=n_rows,
        n_cols=n_cols,
        nnz=int(X.nnz),
        density=round(density, 6),
        sparsity=round(1 - density, 6),
        sparse_mb=round(sparse_bytes / mb, 3),
        dense_mb=round(dense_bytes / mb, 3),
        savings_ratio=round(dense_bytes / sparse_bytes, 1) if sparse_bytes else 0.0,
    )


def describe(r: SparsityReport) -> str:
    return (f"shape {r.n_rows} x {r.n_cols}, {r.sparsity:.4%} zeros, {r.nnz:,} non-zeros\n"
            f"  CSR:   {r.sparse_mb:,.2f} MB\n"
            f"  dense: {r.dense_mb:,.2f} MB\n"
            f"  saving: {r.savings_ratio:,.1f}x")
