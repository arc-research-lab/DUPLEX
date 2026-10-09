from pathlib import Path

import numpy as np

# VCK190 device totals
BRAM_MAX, LUT_MAX, FF_MAX = 2807, 897737, 1797173

DATA_CSV = Path(__file__).with_name("post_route.csv")

FEATURES = ["M", "Nk", "Nw", "D", "M_NK", "M_NW", "M_D", "NK_NW", "NK_D", "NW_D",
            "M_NK_D", "M_NW_D", "NK_NW_D"]

FF_COEF = dict(intercept=9181.8, M=139, Nk=-82.9, Nw=175.5, D=33.7, M_NK=1.1, M_NW=0.36,
               M_D=153.2, NK_NW=99, NK_D=-4.24, NW_D=8, M_NK_D=0.0, M_NW_D=22.8,
               NK_NW_D=-9.4)
LUT_COEF = dict(intercept=5171.4, M=248.96, Nk=-286.6, Nw=80, D=112.4, M_NK=4.3, M_NW=78.3,
                M_D=83.1, NK_NW=223.2, NK_D=3.2, NW_D=-27.2, M_NK_D=0.5, M_NW_D=36.4,
                NK_NW_D=-20)


def _features(M, Nk, Nw, D):
    return dict(M=M, Nk=Nk, Nw=Nw, D=D, M_NK=M * Nk, M_NW=M * Nw, M_D=M * D,
                NK_NW=Nk * Nw, NK_D=Nk * D, NW_D=Nw * D, M_NK_D=M * Nk * D,
                M_NW_D=M * Nw * D, NK_NW_D=Nk * Nw * D)


def _linear(coef, feats):
    return coef["intercept"] + sum(coef[f] * feats[f] for f in FEATURES)


def _pow2ceil(v):
    return 2 ** np.ceil(np.log2(v))


def resources(M_total, N_k, N_w, M_w, D, V, raw_len):
    bram = (N_k * N_w * _pow2ceil(V / 1024 / N_w)          # replicas, banked
            + 2 * M_w * _pow2ceil(raw_len / 1024 / M_w)    # x / y trace buffers
            + M_w * _pow2ceil(V / 1024 / M_w))             # output buffers
    feats = _features(M_total, N_k, N_w, D)
    ff, lut = _linear(FF_COEF, feats), _linear(LUT_COEF, feats)
    util = max(bram / BRAM_MAX, lut / LUT_MAX, ff / FF_MAX)
    return bram, ff, lut, util

def _load(csv_path=DATA_CSV):
    import pandas as pd
    df = pd.read_csv(csv_path)
    for name, val in _features(df["M"], df["Nk"], df["Nw"], df["D"]).items():
        df[name] = val
    return df

def _pct_err(y, pred):
    return (y - pred) / y * 100

def fit(df, y_col, feature_cols=FEATURES, verbose=True):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    d = df.dropna(subset=[y_col]).reset_index(drop=True)
    train, test = d[d["split"] == "train"].copy(), d[d["split"] == "test"].copy()

    Xtr, ytr = train[feature_cols].values.astype(float), train[y_col].values.astype(float)
    reg = LinearRegression().fit(Xtr, ytr)
    train["pred"] = reg.predict(Xtr)
    train["err_pct"] = _pct_err(ytr, train["pred"])

    stats = dict(n_train=len(train), n_test=len(test),
                 r2_train=r2_score(ytr, train["pred"]),
                 mape_train=train["err_pct"].abs().mean())
    if len(test):
        Xte, yte = test[feature_cols].values.astype(float), test[y_col].values.astype(float)
        test["pred"] = reg.predict(Xte)
        test["err_pct"] = _pct_err(yte, test["pred"])
        stats.update(mape_test=test["err_pct"].abs().mean(),
                     max_err_test=test["err_pct"].abs().max())

    if verbose:
        print(f"--- {y_col} ---")
        print(f"  intercept = {reg.intercept_:.3f}")
        for f, c in zip(feature_cols, reg.coef_):
            print(f"  {f:10s} coef = {c:+.4f}")
        print(f"  train: n={stats['n_train']}  R^2={stats['r2_train']:.4f}  "
              f"MAPE={stats['mape_train']:.2f}%")
        if len(test):
            print(f"  TEST (held-out, reported in paper): n={stats['n_test']}  "
                  f"MAPE={stats['mape_test']:.2f}%  max |err|={stats['max_err_test']:.2f}%")
            print(test[["M", "Nk", "Nw", "D", y_col, "pred", "err_pct"]].to_string(
                index=False, formatters={"pred": "{:.0f}".format,
                                         "err_pct": "{:+.1f}".format}))
        print()
    return reg, stats


if __name__ == "__main__":
    data = _load()
    fit(data, "FF")
    fit(data, "LUT")