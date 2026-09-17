"""Point 1 — sparse structures and quantified RAM savings (CSR vs dense)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from scipy.sparse import csr_matrix, issparse


@dataclass
class SparsityReport:
    n_rows: int
    n_cols: int
    nnz: int
    density: float
    sparsity: float
    sparse_bytes: int
    dense_bytes: int
    savings_bytes: int
    savings_ratio: float
    sparse_mb: float
    dense_mb: float

    def as_row(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def csr_footprint_bytes(X: csr_matrix) -> int:
    return int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)


def analyse(X, itemsize: int = 8) -> SparsityReport:
    if not issparse(X):
        X = csr_matrix(X)
    X = X.tocsr()
    n_rows, n_cols = X.shape
    nnz = int(X.nnz)
    total = n_rows * n_cols
    density = (nnz / total) if total else 0.0
    sparse_bytes = csr_footprint_bytes(X)
    dense_bytes = int(total * itemsize)
    return SparsityReport(
        n_rows=n_rows, n_cols=n_cols, nnz=nnz,
        density=round(density, 6), sparsity=round(1.0 - density, 6),
        sparse_bytes=sparse_bytes, dense_bytes=dense_bytes,
        savings_bytes=dense_bytes - sparse_bytes,
        savings_ratio=round(dense_bytes / sparse_bytes, 1) if sparse_bytes else 0.0,
        sparse_mb=round(sparse_bytes / (1024 ** 2), 3),
        dense_mb=round(dense_bytes / (1024 ** 2), 3),
    )


def pretty(report: SparsityReport) -> str:
    return (
        f"shape={report.n_rows}x{report.n_cols}  "
        f"sparsity={report.sparsity:.4%}  nnz={report.nnz:,}\n"
        f"  CSR footprint : {report.sparse_mb:,.3f} MB\n"
        f"  dense equiv.  : {report.dense_mb:,.3f} MB\n"
        f"  RAM saving    : {report.savings_ratio:,.1f}x "
        f"({report.savings_bytes / (1024 ** 2):,.1f} MB saved)"
    )
