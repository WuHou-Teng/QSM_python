import numpy as np


def echofit(ph, mag, TE, inter=0):
    # check ph and mag have same dimensions
    if ph.shape != mag.shape:
        raise ValueError("Input phase and magnitude must have the same size")

    np, nv, ns, ne = ph.shape

    ph = np.transpose(ph, (3, 0, 1, 2)).reshape(ne, -1)
    mag = np.transpose(mag, (3, 0, 1, 2)).reshape(ne, -1)

    if not inter:
        # if assume zero inter
        TE_rep = np.tile(TE[:, np.newaxis], (1, np * nv * ns))

        lfs = np.sum(mag * ph * TE_rep, axis=0) / (np.sum(mag * TE_rep * TE_rep, axis=0) + np.finfo(float).eps)
        lfs = lfs.reshape(np, nv, ns)

        # calculate the fitting residual
        lfs_rep = np.transpose(np.tile(lfs[:, np.newaxis, :, :], (1, ne, 1, 1)), (1, 0, 2, 3))
        res = np.reshape(np.sum((ph - lfs_rep * TE_rep) * mag * (ph - lfs_rep * TE_rep), axis=0) / np.sum(mag, axis=0) * ne, (np, nv, ns))
        res[np.isnan(res)] = 0
        res[np.isinf(res)] = 0

    else:
        # non-zero inter
        x = np.column_stack((TE, np.ones(len(TE))))
        beta = np.zeros((2, np * nv * ns))
        res = np.zeros((np, nv, ns))

        for i in range(np * nv * ns):
            y = ph[:, i]
            w = mag[:, i]
            beta[:, i] = np.linalg.lstsq(x.T @ np.diag(w) @ x, x.T @ np.diag(w) @ y, rcond=None)[0]
            res[i] = (y - x @ beta[:, i]) @ np.diag(w) @ (y - x @ beta[:, i]) / np.sum(w, axis=0) * ne

        beta[np.isnan(beta)] = 0
        beta[np.isinf(beta)] = 0
        res[np.isnan(res)] = 0
        res[np.isinf(res)] = 0

        lfs = np.reshape(beta[0], (np, nv, ns))
        off = np.reshape(beta[1], (np, nv, ns))
        res = np.reshape(res, (np, nv, ns))

    return lfs, res, off
