import numpy as np
from scipy.sparse.linalg import cgs, LinearOperator


def projectionontodipolefields(tfs, mask, vox, weight, z_prjs, num_iter):
    if z_prjs is None:
        z_prjs = np.array([0, 0, 1])  # PURE axial slices

    Nx, Ny, Nz = tfs.shape

    X, Y, Z = np.mgrid[-Nx/2:Nx/2, -Ny/2:Ny/2, -Nz/2:Nz/2]

    X = X * vox[0]
    Y = Y * vox[1]
    Z = Z * vox[2]

    d = (3 * (X * z_prjs[0] + Y * z_prjs[1] + Z * z_prjs[2])**2 - X**2 - Y**2 - Z**2) / (4 * np.pi * (X**2 + Y**2 + Z**2)**2.5)
    d[np.isnan(d)] = 0
    D = np.fft.fftn(np.fft.fftshift(d))

    M = 1 - mask
    W = weight * mask

    b = M * np.fft.ifftn(D * np.fft.fftn(W * W * tfs)).flatten()

    # function y = Afun(x)
    #     x = reshape(x,[Nx,Ny,Nz]);
    #     y = M.*ifftn(D.*fftn(W.*W.*ifftn(D.*fftn(M.*x))));
    #     y = y(:);
    # end

    # res = cgs(@Afun,b,1e-6, num_iter);
    # lfs = mask.*real(tfs-ifftn(D.*fftn(M.*reshape(res,[Nx,Ny,Nz]))));
    def Afun(x):
        x = x.reshape([Nx, Ny, Nz])
        y = M * np.fft.ifftn(D * np.fft.fftn(W * W * np.fft.ifftn(D * np.fft.fftn(M * x))))
        return np.reshape(y, (y.size, 1))

    A = LinearOperator((b.size, b.size), Afun)
    tol = 1e-6
    res = cgs(A, b, tol=tol, maxiter=num_iter)

    lfs = mask * np.real(tfs - np.fft.ifftn(D * np.fft.fftn(M * res.reshape([Nx, Ny, Nz]))))

    bkg_sus = res.reshape([Nx, Ny, Nz])
    bkg_field = mask * np.real(np.fft.ifftn(D * np.fft.fftn(M * bkg_sus)))

    return lfs, bkg_sus, bkg_field
