
import numpy as np

MATRICES = {
    "real-world": np.array([[1.09491743e+00, 8.93391379e-02, -2.11260141e+01],
                            [5.39380112e-02, 1.14434218e+00, -2.90215322e+01],
                            [1.26615050e-04, 2.19446332e-04, 1.00000000e+00]]),
    "rotation":   np.array([[0.866025403784, -0.5, 31.38174251267],        # 30 degrees
                            [0.5, 0.866025403784, -18.11825748733],
                            [0.0, 0.0, 1.0]]),
    "keystone":   np.array([[0.5, -0.25, 24.75],
                            [0.0, 0.5, 0.0],
                            [0.0, -0.005050505051, 1.0]]),
    "zoom_out":   np.array([[0.5, 0.0, 24.75],                              # 0.5x
                            [0.0, 0.5, 24.75],
                            [0.0, 0.0, 1.0]]),
    "general":    np.array([[0.547080290908, -0.048540142203, 9.89999961853],
                            [-0.164051089111, 0.674817575283, 19.799999237061],
                            [-0.002838605099, 0.000294920385, 1.0]]),
}


def perspective_trace(fwd, h=100, w=100, drop_out_of_bounds=False):
    if isinstance(fwd, str):
        fwd = MATRICES[fwd]
    inv = np.linalg.inv(fwd)

    i, j = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64),
                       indexing="ij")
    x_num = inv[0, 0] * j + inv[0, 1] * i + inv[0, 2]
    y_num = inv[1, 0] * j + inv[1, 1] * i + inv[1, 2]
    z_num = inv[2, 0] * j + inv[2, 1] * i + inv[2, 2]

    # int(v + 0.5) truncates toward zero, as in the original C-style loop
    x = np.trunc(x_num / z_num + 0.5).astype(np.int32).ravel()
    y = np.trunc(y_num / z_num + 0.5).astype(np.int32).ravel()

    if drop_out_of_bounds:
        keep = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x, y = x[keep], y[keep]
    return x, y
