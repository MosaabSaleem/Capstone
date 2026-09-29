"""Point 1 — sparse structures and quantified RAM savings (CSR vs dense)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from scipy.sparse import csr_matrix, issparse


@dataclass
class SparsityReport:
    n_rows: int; n_cols: int; nnz: int; density: float; sparsity: float
    sparse_bytes: int; dense_bytes: int; savings_bytes: int
    savings_ratio: float; sparse_mb: float; dense_mb: float

    def as_row(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def csr_footprint_bytes(X) -> int:
    return int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)


def analyse(X, itemsize: int = 8) -> SparsityReport:
    if not issparse(X):
        X = csr_matrix(X)
    X = X.tocsr()
    n_rows, n_cols = X.shape
    nnz = int(X.nnz)
    total = n_rows * n_cols
    density = (nnz / total) if total else 0.0
    sb = csr_footprint_bytes(X)
    db = int(total * itemsize)
    return SparsityReport(n_rows, n_cols, nnz, round(density, 6),
                          round(1.0 - density, 6), sb, db, db - sb,
                          round(db / sb, 1) if sb else 0.0,
                          round(sb / (1024 ** 2), 3), round(db / (1024 ** 2), 3))


def pretty(r: SparsityReport) -> str:
    return (f"shape={r.n_rows}x{r.n_cols}  sparsity={r.sparsity:.4%}  nnz={r.nnz:,}\n"
            f"  CSR footprint : {r.sparse_mb:,.3f} MB\n"
            f"  dense equiv.  : {r.dense_mb:,.3f} MB\n"
            f"  RAM saving    : {r.savings_ratio:,.1f}x")
