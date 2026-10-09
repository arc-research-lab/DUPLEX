from pathlib import Path

import numpy as np

MATRIX_DIR = Path(__file__).with_name("matrices")

MATRICES = {
    "bcsstk08": "bcsstk08.mtx",   # HB/bcsstk08
    "add20":    "add20.mtx",      # Hamm/add20
    "polblogs": "polblogs.mtx",   # Newman/polblogs
}


def spmv_trace(path):
    from scipy.io import mmread

    if not Path(path).is_absolute() and not Path(path).exists():
        path = MATRIX_DIR / MATRICES.get(path, path)
    A = mmread(path).tocsr()
    col_idx = A.indices.astype(np.int32)
    return col_idx, np.zeros_like(col_idx), A.shape[1]
