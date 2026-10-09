import numpy as np

def uniform_trace(img_height=100, img_width=100, seed=0, oob_prob=0.0):
    rng = np.random.default_rng(seed)

    x = rng.integers(0, img_width, size=(img_height, img_width), dtype=np.int32)
    y = rng.integers(0, img_height, size=(img_height, img_width), dtype=np.int32)

    if oob_prob > 0.0:
        oob = rng.random((img_height, img_width)) < oob_prob
        x = np.where(oob, img_width + 1, x)
        y = np.where(oob, img_height + 1, y)

    return x.ravel(), y.ravel()
