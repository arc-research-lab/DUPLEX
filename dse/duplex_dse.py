
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dse.models.duplex_sim import simulate_duplex
from dse.models.resource_model import BRAM_MAX, FF_MAX, LUT_MAX, resources

RESULTS_DIR = Path(__file__).with_name("results")

# ---------------- Experiment settings (as used in the paper) ----------------
L            = 30                        # performance target (reads/cycle), mode A
BUDGET       = 1.0                       # resource budget (peak utilization), mode B
N_TOTALS_A   = (16, 32, 64, 128)         # N_total = N_k * N_w swept in mode A
N_TOTALS_B   = (32, 64, 128, 256, 512)   # N_total swept in mode B
M_W_MULTS    = (1, 2, 4)                 # M_w = N_w * mult
DEPTHS_DSE   = (1, 2, 4, 8, 16)
DEPTHS_SWEEP = (1, 2, 4, 8, 16, 32)
SKIP_NW1     = False                     # True excludes N_w = 1 (pure duplication) from DUPLEX
TARGET_LIMIT = 1.0                       # baselines, mode A: designs must fit the device
RESP_LAT     = 1                         # bank read latency (cycles)
VERBOSE      = True                      # print each new simulation (slow ones are large crossbars)

NP  = "NP"  
NPR = "NR"   
NR  = "NR"    


def _out(name):
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR / name


# ---------------- Simulation ----------------
def fit_trace(a, n):
    """Slice or edge-pad a trace to exactly n entries (the simulator needs a multiple of M_w)."""
    if n <= len(a):
        return a[:n]
    pad = [(0, n - len(a))] + [(0, 0)] * (a.ndim - 1)
    return np.pad(a, pad, mode="edge")


class GroupSim:
    """Simulates one DUPLEX group for one workload, caching results by (M_w, N_w, D).

    A group's throughput does not depend on N_k (perf = throughput * N_k), so designs that
    differ only in N_k reuse the same run.
    """

    def __init__(self, x, y, V, w):
        self.x, self.y, self.V, self.w = x, y, V, w
        self.raw_len = len(x)
        self.cache = {}

    def run(self, M_w, N_w, d):
        """Return (cycles, reads per cycle) for one group."""
        key = (M_w, N_w, d)
        if key not in self.cache:
            t0 = time.time()
            n_this = M_w * ((self.raw_len + M_w - 1) // M_w)
            b_depth = int(np.ceil(self.V / N_w))
            self.cache[key] = simulate_duplex(M_w, N_w, d, RESP_LAT, b_depth, self.w,
                                              self.raw_len, 0, fit_trace(self.x, n_this),
                                              fit_trace(self.y, n_this))
            if VERBOSE:
                print(f"    sim M_w={M_w:<4d} N_w={N_w:<4d} D={d:<3d} {time.time() - t0:7.1f}s")
        return self.cache[key]


def make_sims(workloads, tag, use_cache=True):
    """One GroupSim per workload, pre-loaded from results/simcache_<tag>.csv if it exists."""
    sims = {name: GroupSim(x, y, V, w) for name, (x, y, V, w) in workloads.items()}
    path = _out(f"simcache_{tag}.csv")
    if use_cache and path.exists():
        df = pd.read_csv(path, float_precision="round_trip")
        for r in df.itertuples(index=False):
            if r.workload in sims:
                sims[r.workload].cache[(r.M_w, r.N_w, r.D)] = (int(r.cycles), float(r.rate))
        print(f"Loaded {len(df)} cached simulations from {path.name}")
    return sims


def save_sims(sims, tag):
    """Write every simulation run so far to results/simcache_<tag>.csv."""
    rows = [dict(workload=name, M_w=k[0], N_w=k[1], D=k[2], cycles=int(c), rate=float(r))
            for name, s in sims.items() for k, (c, r) in sorted(s.cache.items())]
    pd.DataFrame(rows).to_csv(_out(f"simcache_{tag}.csv"), index=False)


# ---------------- Design space ----------------
def design_space(n_totals):
    """DUPLEX configurations: (N_total, N_w, N_k, M_w, M_total, D)."""
    for N_total in n_totals:
        for N_w in [2**e for e in range(8) if 2**e <= N_total]:
            if N_total % N_w != 0 or (SKIP_NW1 and N_w == 1):
                continue
            N_k = N_total // N_w
            for mult in M_W_MULTS:
                M_w = N_w * mult
                for d in DEPTHS_DSE:
                    yield N_total, N_w, N_k, M_w, M_w * N_k, d


def baseline_space(n_totals, kind):
    """kind "D" = full duplication (N_w = 1), "X" = full crossbar (N_k = 1)."""
    for N_total in n_totals:
        N_w = 1 if kind == "D" else N_total
        N_k = N_total // N_w
        for mult in M_W_MULTS:
            M_w = N_w * mult
            for d in DEPTHS_DSE:
                yield N_total, N_w, N_k, M_w, M_w * N_k, d


def perf_upper_bound(M_total, N_k, N_w):
    """No design exceeds one read per PE per cycle, or one admitted read per bank per cycle
    (2x margin kept)."""
    return min(M_total, 2 * N_w * N_k)


def print_best(wl_name, best, mode_msg):
    print(f"\n--- DSE Best Point for {wl_name} ---")
    if best is None:
        print(mode_msg)
        return
    print(f"N_total={best.N_total}  M_total={best.M_total}  N_k={best.N_k}  N_w={best.N_w}  "
          f"D={best.D}  perf={best.perf:.2f}  BRAM={best.BRAM:.0f}  FF={best.FF:.0f}  "
          f"LUT={best.LUT:.0f}  max_util={best.max_util:.3f}  "
          f"(BRAM {100*best.BRAM/BRAM_MAX:.1f}%  LUT {100*best.LUT/LUT_MAX:.1f}%  "
          f"FF {100*best.FF/FF_MAX:.1f}%)")


# ---------------- Experiments ----------------
def run_depth_sweep(workloads, configs, tag):
    """Group efficiency (reads / cycle / bank) vs queue depth for fixed (m_w, n_w)."""
    rows = []
    for wl_name, (x, y, V, w) in workloads.items():
        for m_w, n_w in configs:
            b_depth = int(np.ceil(V / n_w))
            n_this = m_w * ((len(x) + m_w - 1) // m_w)
            xp, yp = fit_trace(x, n_this), fit_trace(y, n_this)
            for d in DEPTHS_SWEEP:
                cycles, thr = simulate_duplex(m_w, n_w, d, RESP_LAT, b_depth, w, len(x), 0, xp, yp)
                rows.append(dict(workload=wl_name, m_w=m_w, n_w=n_w, D=d,
                                 cycles=cycles, throughput=thr, eff=thr / n_w))
                print(f"{wl_name:24s} m_w={m_w:<3d} n_w={n_w:<3d} D={d:<3d} "
                      f"cycles={cycles:<10} eff={thr / n_w:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(_out(f"depth_sweep_{tag}.csv"), index=False)
    return df


def run_dse_target(workloads, sims, tag):
    """Mode A: minimize peak utilization subject to perf > L."""
    all_rows, best_rows = [], []
    for wl_name, (x, y, V, w) in workloads.items():
        t0, best = time.time(), None
        for N_total, N_w, N_k, M_w, M_total, d in design_space(N_TOTALS_A):
            cycles, thr = sims[wl_name].run(M_w, N_w, d)
            perf = thr * N_k
            BRAM, FF, LUT, util = resources(M_total, N_k, N_w, M_w, d, V, len(x))
            row = pd.Series(dict(workload=wl_name, N_total=N_total, M_total=M_total, N_k=N_k,
                                 N_w=N_w, M_w=M_w, D=d, perf=perf, BRAM=BRAM, FF=FF, LUT=LUT,
                                 max_util=util, meets_target=perf > L))
            all_rows.append(row)
            if perf > L and (best is None or util < best.max_util):
                best = row
        print_best(wl_name, best, f"No configuration met performance target L > {L}.")
        print(f"   ({time.time() - t0:.1f}s)")
        if best is not None:
            best_rows.append(best)
    pd.DataFrame(all_rows).to_csv(_out(f"dse_target_all_{tag}.csv"), index=False)
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(_out(f"dse_target_best_{tag}.csv"), index=False)
    return best_df


def run_dse_budget(workloads, sims, tag):
    """Mode B: maximize perf subject to peak utilization < BUDGET."""
    all_rows, best_rows = [], []
    for wl_name, (x, y, V, w) in workloads.items():
        t0, best = time.time(), None
        for N_total, N_w, N_k, M_w, M_total, d in design_space(N_TOTALS_B):
            BRAM, FF, LUT, util = resources(M_total, N_k, N_w, M_w, d, V, len(x))
            if util >= BUDGET:
                continue                                  # can't fit: no simulation
            cycles, thr = sims[wl_name].run(M_w, N_w, d)
            perf = thr * N_k
            row = pd.Series(dict(workload=wl_name, N_total=N_total, M_total=M_total, N_k=N_k,
                                 N_w=N_w, M_w=M_w, D=d, perf=perf, BRAM=BRAM, FF=FF, LUT=LUT,
                                 max_util=util))
            all_rows.append(row)
            if best is None or perf > best.perf:
                best = row
        print_best(wl_name, best, f"No configuration fits within max_util < {BUDGET}.")
        print(f"   ({time.time() - t0:.1f}s)")
        if best is not None:
            best_rows.append(best)
    pd.DataFrame(all_rows).to_csv(_out(f"dse_budget_all_{tag}.csv"), index=False)
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(_out(f"dse_budget_best_{tag}.csv"), index=False)
    return best_df


# ---------------- Baselines ----------------
def best_baseline(sim, V, raw_len, kind, mode):
    """Best full-duplication ("D") or full-crossbar ("X") design.

    Returns (perf, util, (M_total, N_k, N_w, D)), or NP / NPR / NR. Selects the same design as
    an exhaustive search, but skips simulations that cannot change the answer.
    """
    if mode == "budget":
        best = None
        for N_total, N_w, N_k, M_w, M_total, d in baseline_space(N_TOTALS_B, kind):
            util = resources(M_total, N_k, N_w, M_w, d, V, raw_len)[3]
            if util >= BUDGET:
                continue                                  # can't fit within B
            if best is not None and perf_upper_bound(M_total, N_k, N_w) <= best[0]:
                continue                                  # can't beat the current best
            perf = sim.run(M_w, N_w, d)[1] * N_k
            if best is None or perf > best[0]:
                best = (perf, util, (M_total, N_k, N_w, d))
        return NR if best is None else best

    # ---- mode A: minimize utilization subject to perf > L ----
    fits, oversized = [], []
    for N_total, N_w, N_k, M_w, M_total, d in baseline_space(N_TOTALS_A, kind):
        if perf_upper_bound(M_total, N_k, N_w) <= L:
            continue                                      # provably can't reach L
        util = resources(M_total, N_k, N_w, M_w, d, V, raw_len)[3]
        (fits if util < TARGET_LIMIT else oversized).append(
            (util, N_total, N_w, N_k, M_w, M_total, d))

    for util, N_total, N_w, N_k, M_w, M_total, d in sorted(fits):
        perf = sim.run(M_w, N_w, d)[1] * N_k
        if perf > L:
            return perf, util, (M_total, N_k, N_w, d)

    deepest = {}
    for util, N_total, N_w, N_k, M_w, M_total, d in oversized:
        key = (N_total, M_w)
        if key not in deepest or d > deepest[key][-1]:
            deepest[key] = (N_w, N_k, M_w, d)
    for N_w, N_k, M_w, d in sorted(deepest.values(), key=lambda v: v[2] * v[0]):
        if sim.run(M_w, N_w, d)[1] * N_k > L:
            return NPR
    return NP


def tex_ratio(r):
    return f"{r:.2f}$\\times$" if isinstance(r, float) else r


def run_baselines(workloads, sims, target_best, budget_best, tag):
    t_idx = target_best.set_index("workload") if len(target_best) else None
    b_idx = budget_best.set_index("workload") if len(budget_best) else None

    rows = []
    for name, (x, y, V, w) in workloads.items():
        t = t_idx.loc[name] if t_idx is not None and name in t_idx.index else None
        b = b_idx.loc[name] if b_idx is not None and name in b_idx.index else None
        row = {"workload": name}
        tex_red, tex_spd = [], []
        for kind in ("D", "X"):
            t0 = time.time()
            bt = best_baseline(sims[name], V, len(x), kind, "target")
            bb = best_baseline(sims[name], V, len(x), kind, "budget")
            print(f"{name:18s} {kind}: done in {time.time() - t0:.1f}s")

            if isinstance(bt, str):
                red = bt
            elif t is None:
                red = "n/a"                            
            else:
                red = round(bt[1] / t.max_util, 2)

            if isinstance(bb, str):
                spd = bb
            elif b is None:
                spd = "n/a"
            else:
                spd = round(b.perf / bb[0], 2)

            row[f"Red.{kind}"] = red
            row[f"{kind}_cfg_T"] = bt if isinstance(bt, str) else bt[2]
            row[f"Spd.{kind}"] = spd
            row[f"{kind}_rate_B"] = bb if isinstance(bb, str) else round(bb[0], 2)
            row[f"{kind}_cfg_B"] = bb if isinstance(bb, str) else bb[2]
            tex_red.append(tex_ratio(red))
            tex_spd.append(tex_ratio(spd))
        row["tex_Red_KW"] = "\\,/\\,".join(tex_red)
        row["tex_Spd_KW"] = "\\,/\\,".join(tex_spd)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(_out(f"baselines_{tag}.csv"), index=False)
    return df
