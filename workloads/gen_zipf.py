import numpy as np

def zipf_trace(total_iters, img_width=100, img_height=100, skew=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n_addrs = img_width * img_height

    if skew == 0.0:
        probs = np.full(n_addrs, 1.0 / n_addrs)
    else:
        ranks = np.arange(1, n_addrs + 1)
        weights = 1.0 / (ranks ** skew)
        probs = weights / weights.sum()

    perm = rng.permutation(n_addrs)
    rank_samples = rng.choice(n_addrs, size=total_iters, p=probs)
    addr = perm[rank_samples]

    x = (addr % img_width).astype(np.int64)
    y = (addr // img_width).astype(np.int64)
    return x, y
