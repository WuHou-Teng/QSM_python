import numpy as np


def echofit(ph, mag, TE, inter=0):
    # check ph and mag have same dimensions
    if ph.shape != mag.shape:
        raise ValueError("Input phase and magnitude must have the same size")

    npp, nv, ns, ne = ph.shape

    ph = np.transpose(ph, (3, 0, 1, 2)).reshape(ne, -1)
    mag = np.transpose(mag, (3, 0, 1, 2)).reshape(ne, -1)

    if not inter:
        # if assume zero inter
        TE_rep = np.tile(TE[:, np.newaxis], (1, npp * nv * ns))
        # TE_rep = repmat(TE(:), [1 np * nv * ns]);
        # TE_rep = np.tile(TE, (5, npp * nv * ns))

        lfs = np.sum(mag * ph * TE_rep, axis=0) / (np.sum(mag * TE_rep * TE_rep, axis=0) + np.finfo(float).eps)
        lfs = lfs.reshape(npp, nv, ns)

        # calculate the fitting residual
        # lfs_rep = permute(repmat(lfs(:), [1 ne]), [2 1]);
        lfs_rep = np.transpose(np.tile(lfs.flatten(), (ne, 1)))
        lfs_rep = lfs_rep.T
        res = np.reshape(np.sum((ph - lfs_rep * TE_rep) * mag * (ph - lfs_rep * TE_rep), axis=0) / np.sum(mag, axis=0) * ne, (npp, nv, ns))
        res[np.isnan(res)] = 0
        res[np.isinf(res)] = 0
        off = 0

    else:
        # non-zero inter
        x = np.column_stack((TE, np.ones(len(TE))))
        beta = np.zeros((2, npp * nv * ns))
        res = np.zeros((npp, nv, ns))

        for i in range(npp * nv * ns):
            y = ph[:, i]
            w = mag[:, i]
            beta[:, i] = np.linalg.lstsq(x.T @ np.diag(w) @ x, x.T @ np.diag(w) @ y, rcond=None)[0]
            res[i] = (y - x @ beta[:, i]) @ np.diag(w) @ (y - x @ beta[:, i]) / np.sum(w, axis=0) * ne

        beta[np.isnan(beta)] = 0
        beta[np.isinf(beta)] = 0
        res[np.isnan(res)] = 0
        res[np.isinf(res)] = 0

        lfs = np.reshape(beta[0], (npp, nv, ns))
        off = np.reshape(beta[1], (npp, nv, ns))
        res = np.reshape(res, (npp, nv, ns))

    return lfs, res, off
