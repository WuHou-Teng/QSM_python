import numpy as np
from scipy.ndimage import convolve


def sharp(tfs, mask, vox=None, ker_rad=None, tsvd=None):
    if vox is None:
        vox = [1, 1, 1]

    if ker_rad is None:
        ker_rad = 4

    if tsvd is None:
        tsvd = 0.05

    imsize = tfs.shape

    # make spherical/ellipsoidal convolution kernel (ker)
    rx = round(ker_rad / vox[0])
    ry = round(ker_rad / vox[1])
    rz = round(ker_rad / vox[2])
    rx = max(rx, 1)
    ry = max(ry, 1)
    rz = max(rz, 1)
    [X, Y, Z] = np.mgrid[-rx:rx+1, -ry:ry+1, -rz:rz+1]
    h = (X**2 / rx**2 + Y**2 / ry**2 + Z**2 / rz**2 <= 1).astype(float)
    ker = h / np.sum(h)

    # erode the mask by convolving with the kernel
    mask_ero = np.zeros(imsize)
    mask_tmp = convolve(mask, ker)
    mask_ero[mask_tmp > 1 - 1 / np.sum(h)] = 1

    # prepare convolution kernel: delta-ker
    dker = -ker
    dker[rx, ry, rz] = 1 - ker[rx, ry, rz]
    DKER = np.fft.fftn(dker, imsize)

    # SHARP
    # convolute the total field (ext + int) with d_kernel
    ph_tmp = np.fft.ifftn(np.fft.fftn(tfs) * DKER)
    csh = np.array([rx, ry, rz])
    ph_tmp = np.roll(ph_tmp, np.negative(csh), axis=(0, 1, 2))
    # erode the result (abandon brain edges)
    ph_tmp *= mask_ero
    # deconvolution
    ph_int = np.fft.ifftn(np.fft.fftn(ph_tmp) / DKER)
    ph_int[np.abs(DKER) < tsvd] = 0
    ph_tmp = np.roll(ph_int, csh, axis=(0, 1, 2))
    lfs = np.real(ph_tmp) * mask_ero

    return lfs, mask_ero
