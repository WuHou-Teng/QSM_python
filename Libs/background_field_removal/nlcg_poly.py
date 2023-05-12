import numpy as np


# from Libs.dipole_inversion.nlcg import wGradient, objFunc


# function [m,RES] = nlcg_poly(m0,params)
def nlcg_poly(m0, params):
    """
    Phi(m) = ||W(Fu*m - y)||^2 + lamda1*|TV*m|_1
    m:       susceptibility
    W:       weighting matrix derived from magnitude intensities
    Fu:      F_{-1}*D*F forward calculates the field from susceptibility
    y:       measured field to be fitted (inversion)
    lambda1: TV regularization parameter
    TV:      total variation operation
    ||...||^2: L2 norm
    |...|_1: L1 norm
    note the TV term can also be L2 norm if set p=2,
    then the term would be changed to ||TV*m||^2
    :param m0:
    :param params:
    :return:
    """

    m = m0

    # % line search parameters
    # maxlsiter = params.lineSearchItnlim;
    # gradToll  = params.gradToll;
    # alpha     = params.lineSearchAlpha;
    # beta      = params.lineSearchBeta;
    # t0        = params.lineSearchT0;
    # k         = 0;
    maxlistiter = params.lineSearchItnlim
    gradToll = params.gradToll
    alpha = params.lineSearchAlpha
    beta = params.lineSearchBeta
    t0 = params.lineSearchT0
    k = 0

    # % compute (-gradient): search direction
    g0 = wGradient(m, params)
    dm = -g0

    f = 0

    # % iterations
    # while(k <= params.Itnlim)
    while k <= params.Itnlim:
        # % backtracking line-search
        t = t0
        #     t = t0;
        # f0 = objFunc(m,dm,0,params);
        # f1 = objFunc(m,dm,t,params);
        # lsiter = 0;
        f0 = objFunc(m, dm, 0, params)
        f1 = objFunc(m, dm, t, params)
        lsiter = 0
        # while (f1 > f0 + alpha*t*(g0(:)'*dm(:))) && (lsiter<maxlsiter)
        #     t = t* beta;
        #     f1 = objFunc(m,dm,t,params);
        #     lsiter = lsiter + 1;
        # end
        while (f1 > f0 + alpha * t * (g0.ravel() @ dm.ravel())) and (lsiter < maxlistiter):
            t = t * beta
            f1 = objFunc(m, dm, t, params)
            lsiter = lsiter + 1
        # % control the number of line searches by adapting the initial step search
        # if lsiter > 2
        #     t0 = t0 * beta;
        # end
        # if lsiter < 1
        #     t0 = t0 / beta;
        # end
        if lsiter > 2:
            t0 = t0 * beta
        if lsiter < 1:
            t0 = t0 / beta

        # % updates
        m = m + t * dm
        dm0 = dm
        g1 = wGradient(m, params)
        # bk = g1(:)'*g1(:)/(g0(:)'*g0(:)+eps)
        bk = (g1.ravel() @ g1.ravel()) / (g0.ravel() @ g0.ravel() + np.finfo(float).eps)
        g0 = g1
        dm = -g1 + bk * dm
        k = k + 1

        # % outputs for debugging purpose
        # fprintf('%d , relative changes: %f\n', k, norm(t*dm(:))/norm(m(:)));
        print(f'{k}, relative changes: {np.linalg.norm(t * dm.ravel()) / np.linalg.norm(m.ravel())}')

        # if (norm(t*dm(:))/norm(m(:)) <= gradToll);
        #     count = count + 1;
        # else
        #     count = 0;
        # end
        count = 0
        if np.linalg.norm(t * dm.ravel()) / np.linalg.norm(m.ravel()) <= gradToll:
            count = count + 1
        else:
            count = 0
        # if (count == 10)
        #     break;
        # end
        if count == 10:
            break

        f = f1

    return m


# function obj = objFunc(m, dm, t, params)
# w1 = m+t*dm;
# RES = exp(1j*params.P*w1) - params.data;
# obj = RES(:)'*RES(:) + params.lambda*w1'*w1;
def objFunc(m, dm, t, params):
    w1 = m + t * dm
    RES = np.exp(1j * params.P @ w1) - params.data
    obj = RES.ravel() @ RES.ravel() + params.lambda1 * w1.ravel() @ w1.ravel()
    return obj


# function grad = wGradient(m,params)
#
# % construct the diagonal matrix
#
# grad = (-1j*params.P.'.*repmat(exp(-1j*params.P*m).',[size(params.P,2),1]))*(exp(1j*params.P*m)-params.data) +
#        (1j*params.P.'.*repmat(exp(1j*params.P*m).',[size(params.P,2),1])) *
#        conj(exp(1j*params.P*m)-params.data) + 2*params.lambda*m;
def wGradient(m, params):
    # construct the diagonal matrix
    grad = ((-1j * params.P.T * np.tile(np.exp(-1j * params.P @ m).T, (params.P.shape[1], 1))) *
            (np.exp(1j * params.P @ m) - params.data) +
            (1j * params.P.T * np.tile(np.exp(1j * params.P @ m).T, (params.P.shape[1], 1))) *
            np.conj(np.exp(1j * params.P @ m) - params.data) + 2 * params.lambda1 * m)
    return grad
