from .gen_perspective import perspective_trace
from .gen_spmv import spmv_trace
from .gen_uniform import uniform_trace
from .gen_zipf import zipf_trace

H, W = 100, 100
V_IMG = H * W
ZIPF_ITERS = H * W


def _uniform():
    x, y = uniform_trace(H, W, seed=0)
    return x, y, V_IMG, W


def _zipf(s):
    x, y = zipf_trace(ZIPF_ITERS, W, H, skew=s, seed=0)
    return x, y, V_IMG, W


def _spmv(name):
    x, y, V = spmv_trace(name)
    return x, y, V, V          # y = 0, so the address is the column index for any w


def _persp(name):
    x, y = perspective_trace(name, H, W)
    return x, y, V_IMG, W


SUITES = {
    "uniform": {"Uniform": _uniform},
    "zipf": {f"Zipf s={s}": (lambda s=s: _zipf(s)) for s in (0.5, 1.0, 1.5)},
    "spmv": {f"SpMV ({m})": (lambda m=m: _spmv(m))
             for m in ("power", "bcsstk08", "add20", "polblogs")},
    "perspective": {f"Persp ({m})": (lambda m=m: _persp(m))
                    for m in ("real-world", "rotation", "zoom_out")},
}


def load_suite(suite):
    """All workloads of one suite, as {name: (x, y, V, w)}."""
    return {name: load() for name, load in SUITES[suite].items()}


def load_workload(name):
    """One workload by display name, e.g. "SpMV (add20)"."""
    for suite in SUITES.values():
        if name in suite:
            return suite[name]()
    raise KeyError(name)
