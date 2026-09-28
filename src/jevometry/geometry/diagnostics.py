"""Matrix diagnostics for Fisher information and derived matrices."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DEFAULT_RANK_RTOL = 1e-8
DEFAULT_RANK_ATOL = 1e-12


@dataclass
class MatrixDiagnostics:
    """Eigen-decomposition, rank and conditioning of a square matrix."""

    symmetry_residual: float
    eigenvalues: FloatArray
    eigenvectors: FloatArray
    rank: int
    condition_number: float | None
    null_directions: list[FloatArray] = field(default_factory=list)
    psd_violation: bool = False
    min_eigenvalue: float = 0.0
    max_eigenvalue: float = 0.0
    rank_rtol: float = DEFAULT_RANK_RTOL
    rank_atol: float = DEFAULT_RANK_ATOL
    diagnostics: list[str] = field(default_factory=list)


def matrix_diagnostics(
    matrix: FloatArray,
    *,
    rtol: float = DEFAULT_RANK_RTOL,
    atol: float = DEFAULT_RANK_ATOL,
) -> MatrixDiagnostics:
    """Compute symmetry, spectrum, rank, conditioning and null directions.

    The raw matrix is never projected to PSD.  Negative eigenvalues from
    rounding are reported as-is.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix diagnostics require a square matrix")
    scale = max(float(np.linalg.norm(array, ord="fro")), atol)
    symmetry_residual = float(np.linalg.norm(array - array.T, ord="fro") / scale)
    symmetrized = 0.5 * (array + array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrized)
    singular_values = np.linalg.svd(array, compute_uv=False)
    if singular_values.size == 0:
        rank = 0
        condition: float | None = None
    else:
        largest = float(singular_values[0])
        threshold = max(rtol * largest, atol)
        rank = int(np.sum(singular_values > threshold))
        smallest = float(singular_values[-1])
        condition = None if rank < array.shape[0] else largest / smallest
    _, _, right_vectors = np.linalg.svd(array)
    null_directions: list[FloatArray] = []
    if singular_values.size and singular_values.size == right_vectors.shape[0]:
        threshold = max(rtol * float(singular_values[0]), atol)
        for index in range(singular_values.size - 1, -1, -1):
            if singular_values[index] > threshold:
                break
            null_directions.append(right_vectors[index].copy())
    min_eigenvalue = float(eigenvalues.min()) if eigenvalues.size else 0.0
    max_eigenvalue = float(eigenvalues.max()) if eigenvalues.size else 0.0
    messages: list[str] = []
    if symmetry_residual > 1e-10:
        messages.append(f"asymmetry residual {symmetry_residual:.3e}")
    if min_eigenvalue < -atol:
        messages.append(
            f"negative eigenvalue {min_eigenvalue:.3e} exceeds absolute tolerance {atol:.3e}"
        )
    return MatrixDiagnostics(
        symmetry_residual=symmetry_residual,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        rank=rank,
        condition_number=condition,
        null_directions=null_directions,
        psd_violation=min_eigenvalue < -atol,
        min_eigenvalue=min_eigenvalue,
        max_eigenvalue=max_eigenvalue,
        rank_rtol=rtol,
        rank_atol=atol,
        diagnostics=messages,
    )


def is_psd(matrix: FloatArray, *, atol: float = 1e-12) -> bool:
    """PSD check used by the data-processing inequality acceptance test."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.size == 0:
        return True
    eigenvalues = np.linalg.eigvalsh(0.5 * (array + array.T))
    return bool(eigenvalues.min() >= -atol)
