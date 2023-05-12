import numpy as np

from Libs.background_field_removal.nlcg_poly import nlcg_poly
from tools.StructureCLass import Param
from Libs.Misc.NIFTI_python.nifti_base import make_save_nii_engine


# % nonlinear polynomial fit using nonlinear CG
# function polyfit_nonlinear = poly3d_nonlinear(offsets,mask,poly_order)
def poly3d_nonlinear(offsets, mask, poly_order=1):
    # [np nv nv2] = size(offsets);
    npp, nv, nv2 = np.shape(offsets)

    # % polyfit
    # [np nv nv2] = size(lfs);
    npp, nv, nv2 = np.shape(offsets)
    # % polyfit
    px = np.tile(np.arange(1, npp + 1), (nv * nv2, 1))
    px = np.reshape(px, (np.size(px), 1))

    py = np.tile(np.arange(1, nv + 1), (np, 1))
    py = np.tile(np.reshape(py, (np.size(py), 1)), (nv2, 1))
    pz = np.tile(np.arange(1, nv2 + 1), (np * nv, 1))
    pz = np.reshape(pz, (np.size(pz), 1))

    # % fit only the non-zero region
    px_nz = px[np.where(mask)]
    py_nz = py[np.where(mask)]
    pz_nz = pz[np.where(mask)]

    # if poly_order == 1
    # 	% first order polyfit
    # 	P = [px, py, pz, ones(length(px),1)]; % polynomials
    # end
    if poly_order == 1:
        # first order polyfit
        P = np.concatenate((px, py, pz, np.ones((np * nv * nv2, 1))), axis=1)

    # I = offsets(logical(mask)); % measurements of non-zero region
    I = offsets[np.where(mask)]
    # I = I(:);
    I = np.reshape(I, (np.size(I), 1))

    # % start non-linear CG method
    params = Param()
    params.Itnlim = 200  # interations numbers (adjust accordingly!)
    params.gradToll = 1e-4  # step size tolerance stopping criterea
    params.lineSearchItnlim = 200
    params.lineSearchAlpha = 0.01
    params.lineSearchBeta = 0.6
    params.lineSearchT0 = 1  # step size to start with
    params.data = I
    params.P = P  # 此处的P必须当poly_order=1时才能使用
    # % params.wt = 1; % weighting matrix
    # params.lambda = 1e6;
    params.lamb = 1e6  # 原文中的lambda是matlab的关键字，这里改为lamb

    # coeff = nlcg_poly(zeros(4,1), params);
    coeff = nlcg_poly(np.zeros((4, 1)), params)
    #
    # % name the phase result after polyfit as tfs (total field shift)
    # polyfit_nonlinear = zeros(np*nv*nv2,1);
    polyfit_nonlinear = np.zeros((np * nv * nv2, 1))
    # polyfit_nonlinear(logical(mask(:))) = exp(1j*P*coeff);
    polyfit_nonlinear[np.where(mask)] = np.exp(1j * np.dot(P, coeff))
    # polyfit_nonlinear = reshape(polyfit_nonlinear,[np,nv,nv2]);
    polyfit_nonlinear = np.reshape(polyfit_nonlinear, (np, nv, nv2))

    # nii = make_nii(angle(polyfit_nonlinear));
    # save_nii(nii,'offsets_poly.nii')
    make_save_nii_engine(np.angle(polyfit_nonlinear), "polyfit_nonlinear", [1, 1, 1], 'offsets_poly.nii')
