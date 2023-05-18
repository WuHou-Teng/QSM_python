import numpy as np

from Libs.dipole_inversion.nlcg import nlcg
from Libs.dipole_inversion.cls_tv import Cls_TV
from Libs.dipole_inversion.cls_dipconv import Cls_Dipconv
from Transfer_Tools.numpy_over_write import padarray, ma_range, nd_grid_original
from tools.StructureCLass import Param
from Libs.Misc.NIFTI_python.nifti_base import make_save_nii_engine


def tvdi(lfs, mask, vox, tv_reg, weights, z_prjs=None, Itnlim=None, pNorm=None):
    """
    TVDI Total variation dipole inversion.
    Method is similar to Appendix in the following paper
        <Lustig, M., Donoho, D. and Pauly, J. M. (2007)>
    Sparse MRI: 		The application of compressed sensing for rapid MR imaging.
    Magn Reson Med, 58: 1182–1195. doi: 10.1002/mrm.21391

    [SUS,RES] = TVDI(LFS,MASK,VOX,TV_REG,WEIGHTS,Z_PRJS,ITNLIM,PNORM)

    :param lfs: local field shift (field perturbation map)
    :param mask: binary mask defining ROI
    :param vox: voxel size, e.g. [1 1 1] for isotropic resolution
    :param tv_reg: Total Variation regularization paramter, e.g. 5e-4
    :param weights: weights for the data consistancy term, e.g. mask or magnitude
    :param z_prjs: normal vector of the imaging plane, e.g. [0,0,1]
    :param Itnlim: interation numbers of nlcg, e.g. 500
    :param pNorm: L1 or L2 norm regularization
    :return:
        SUS : susceptibility distribution after dipole inversion
        RES : residual field after QSM fitting
    """

    # pad extra 20 slices on both sides
    lfs = padarray(lfs, [0, 0, 20])
    mask = padarray(mask, [0, 0, 20])
    weights = padarray(weights, [0, 0, 20])

    z_prjs = [0, 0, 1] if z_prjs is None else z_prjs  # PURE axial slices

    Itnlim = 500 if Itnlim is None else Itnlim

    pNorm = 1 if pNorm is None else pNorm

    Nx, Ny, Nz = np.shape(lfs)
    imsize = np.shape(lfs)

    # weights for data consistancy term (normalized)
    W = mask * weights
    W = W/np.sum(W[:]) * np.sum(mask[:])
    # to be consistent with tfi_nlcg.m
    # W = weights/sqrt(sum(weights(:).^2)/numel(weights));

    # % set the DC point of field in k-space to 0
    # % mean value of lfs to be 0
    # lfs = lfs.*mask;
    # lfs = lfs-sum(lfs(:))/sum(mask(:));
    # lfs = lfs.*mask;

    # create K-space filter kernel D
    FOV = np.array(vox) * np.array([Nx, Ny, Nz])
    try:
        FOVx = FOV[0]
        FOVy = FOV[1]
        FOVz = FOV[2]
    except IndexError:
        # FOC might be a 1x3 array rather than a 0x3 array
        FOVx = FOV[0][0]
        FOVy = FOV[0][1]
        FOVz = FOV[0][2]

    x = ma_range(-Nx/2, Nx/2-1)
    y = ma_range(-Ny/2, Ny/2-1)
    z = ma_range(-Nz/2, Nz/2-1)

    # indexing='ij' 输出维度为MNP，而indexing='xy'输出维度为NMP
    kx, ky, kz = np.array(nd_grid_original(x/FOVx, y/FOVy, z/FOVz, indexing='ij'))
    # D = 1/3 - kz.^2./(kx.^2 + ky.^2 + kz.^2);
    D = 1/3 - ((kx * z_prjs[0] + ky * z_prjs[1] + kz * (z_prjs[2]))**2)/(kx**2 + ky**2 + kz**2)
    # D(floor(Nx/2+1),floor(Ny/2+1),floor(Nz/2+1)) = 0
    D[Nx//2, Ny//2, Nz//2] = 0
    D = np.fft.fftshift(D)

    # Matlab中的结构体用class代替
    param = Param()
    # parameter structures for inversion
    # data consistancy and TV term objects
    param.FT = Cls_Dipconv([Nx, Ny, Nz], D)
    param.TV = Cls_TV()

    param.Itnlim = Itnlim  # interations numbers (adjust accordingly!)
    param.gradToll = 1e-4  # step size tolerance stopping criterea
    # TODO: 我不确定这里是否应该这么写
    param.l1Smooth = np.finfo(float).eps  # eps, 1e-15; smoothing parameter of L1 norm
    param.pNorm = pNorm  # type of norm to use (i.e. L1 L2 etc)
    param.lineSearchItnlim = 100
    param.lineSearchAlpha = 0.01
    param.lineSearchBeta = 0.6
    param.lineSearchT0 = 1   # step size to start with

    param.TVWeight = tv_reg  # TV penalty
    # param.mask = mask; %%% not used in nlcg
    param.data = lfs * mask

    param.wt = W  # weighting matrix

    # non-linear conjugate gradient method
    sus, RES_nlcg, TVterm_nlcg = nlcg(np.zeros((Nx, Ny, Nz)), param)

    # if want to keep the dipole fitting result
    # don't mask it, instead, use the following:
    # sus = real(sus).*mask;
    sus = np.real(sus)

    # residual difference between fowardly calculated field and lfs
    res = lfs - np.real(np.fft.ifftn(D * np.fft.fftn(sus)))

    # remove the extra padding slices
    sus = sus[:, :, 20:-1-19]
    res = res[:, :, 20:-1-19]
    return sus, res


if __name__ == "__main__":
    from scipy.io import loadmat

    data_source_file = "./Source_and_output/lfs.mat"
    lfs = np.array(loadmat(data_source_file)["lfs"])
    source_file = "./Source_and_output/tfs_and_mask.mat"
    mask = np.array(loadmat(source_file)["mask"])
    vox = [1, 1, 1]
    tv_reg = 5e-4
    weights = mask
    z_prjs = [0, 0, 1]
    Itnlim = 500
    pNorm = 1
    sus, res = tvdi(lfs, mask, vox, tv_reg, weights, z_prjs, Itnlim, pNorm)
    make_save_nii_engine(sus, "sus", [1, 1, 1], "sus.nii")
    make_save_nii_engine(res, "res", [1, 1, 1], "res.nii")

