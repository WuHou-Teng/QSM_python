import numpy as np


def nlcg_tik(m0, params):
    # Parameters
    maxlsiter = params.lineSearchItnlim
    gradToll = params.gradToll
    alpha = params.lineSearchAlpha
    beta = params.lineSearchBeta
    t0 = params.lineSearchT0

    # Initialize variables
    m = m0
    g0 = wGradient(m, params)
    dm = -g0
    f0 = 0
    count = 0
    k = 0

    # Iterations
    while k <= params.Itnlim:
        # Backtracking line search
        t = t0
        f1, Res_term, TV_term, Tik_term, TV_term2 = objFunc(m, dm, t, params)
        lsiter = 0
        while f1 > f0 + alpha * t * (np.dot(g0.flatten(), dm.flatten())):
            t = t * beta
            f1, Res_term, TV_term, Tik_term, TV_term2 = objFunc(m, dm, t, params)
            lsiter += 1

        # Control the number of line searches
        if lsiter > 2:
            t0 = t0 * beta
        if lsiter < 1:
            t0 = t0 / beta

        # Updates
        m += t * dm
        dm0 = dm
        g1 = wGradient(m, params)
        bk = np.dot(g1.flatten(), g1.flatten()) / (np.dot(g0.flatten(), g0.flatten()) + np.finfo(float).eps)
        g0 = g1
        dm = -g1 + bk * dm
        k += 1

        print('.')

        if np.linalg.norm(t * dm.flatten()) / np.linalg.norm(m.flatten()) <= gradToll:
            count += 1
        else:
            count = 0

        if count == 10:
            break

    return m, Res_term, TV_term, Tik_term


# function [obj,Res_term,TV_term,Tik_term,TV_term2] = objFunc(m, dm, t, params)
    # p = params.pNorm;
    # w1 = m+t*dm;
    #
    # w2 = params.TV*(params.P.*w1.*params.TV_mask);
    # TV = (w2.*conj(w2)+params.l1Smooth).^(p/2);
    # TV_term = sum(TV(:));
    #
    # % add air TV
    # w3 = params.TV*(params.P.*w1.*params.air_mask);
    # TV2 = (w3.*conj(w3)+params.l1Smooth).^(p/2);
    # TV_term2 = sum(TV2(:));
    #
    # Res_term = params.FT*(params.P.*w1.*params.sus_mask) - params.data;
    # Res_term = (params.Res_wt(:).*Res_term(:))'*(params.Res_wt(:).*Res_term(:));
    #
    # Tik_term = (params.P(:).*params.Tik_mask(:).*w1(:))'*(params.P(:).*params.Tik_mask(:).*w1(:));
    #
    # % obj = Res_term + params.Tik_reg*Tik_term + params.TV_reg*TV_term;
    # obj = Res_term + params.Tik_reg*Tik_term + params.TV_reg*TV_term + params.TV_reg2*TV_term2;
def objFunc(m, dm, t, params):
    p = params.pNorm
    w1 = m + t * dm

    w2 = params.TV.mtimes(params.P * w1 * params.TV_mask)
    TV = (w2 * np.conjugate(w2) + params.l1Smooth) ** (p / 2)
    TV_term = np.sum(TV)

    # add air TV
    w3 = params.TV.mtimes(params.P * w1 * params.air_mask)
    TV2 = (w3 * np.conjugate(w3) + params.l1Smooth) ** (p / 2)
    TV_term2 = np.sum(TV2)

    Res_term = params.FT.mtimes(params.P * w1 * params.sus_mask) - params.data
    Res_term = (np.reshape(params.Res_wt, (np.size(params.Res_wt), 1)) *
                np.reshape(Res_term, (np.size(Res_term), 1))).T.dot(
        np.reshape(params.Res_wt, (np.size(params.Res_wt), 1)) * np.reshape(Res_term, (np.size(Res_term), 1))
    )

    Tik_term = (np.reshape(params.P, (np.size(params.P), 1)) *
                np.reshape(params.Tik_mask, (np.size(params.Tik_mask), 1)) *
                np.reshape(w1, (np.size(w1), 1))).T.dot(
        np.reshape(params.P, (np.size(params.P), 1)) *
        np.reshape(params.Tik_mask, (np.size(params.Tik_mask), 1)) *
        np.reshape(w1, (np.size(w1), 1))
    )

    # obj = Res_term + params.Tik_reg*Tik_term + params.TV_reg*TV_term;
    obj = Res_term + params.Tik_reg * Tik_term + params.TV_reg * TV_term + params.TV_reg2 * TV_term2
    return obj, Res_term, TV_term, Tik_term, TV_term2


# function grad = wGradient(m,params)
    # p = params.pNorm;
    # w1 = params.TV*(params.P.*m.*params.TV_mask);
    # w2 = params.TV*(params.P.*m.*params.air_mask);
    # grad_TV = params.P.*params.TV_mask.*(params.TV'*(p*w1.*(w1.*conj(w1)+params.l1Smooth).^(p/2-1)));
    # grad_TV2 = params.P.*params.air_mask.*(params.TV'*(p*w2.*(w2.*conj(w2)+params.l1Smooth).^(p/2-1)));
    # grad_Res = params.P.*params.sus_mask.*(
    #       params.FT'*(
#               (params.Res_wt.^2).*(
#                   (
#                       params.FT*(
#                           params.P.*params.sus_mask.*m
#                       )
#                   )-params.data
#               )
#          )
#     );
    # grad_Tik = params.P.^2.*params.Tik_mask.^2.*m;
    # grad = 2*grad_Res + params.TV_reg*grad_TV + 2*params.Tik_reg.*grad_Tik + params.TV_reg2*grad_TV2;
def wGradient(m, params):
    p = params.pNorm
    w1 = params.TV.mtimes(params.P * m * params.TV_mask)
    w2 = params.TV.mtimes(params.P * m * params.air_mask)
    grad_TV = params.P * params.TV_mask * (params.TV.trans.mtimes(
        p * w1 * (w1 * np.conj(w1) + params.l1Smooth) ** (p / 2 - 1))
    )
    grad_TV2 = params.P * params.air_mask * (params.TV.trans.mtimes(
        p * w2 * (w2 * np.conj(w2) + params.l1Smooth) ** (p / 2 - 1))
    )
    grad_Res = params.P * params.sus_mask * (params.FT.trans.mtimes(
        (params.Res_wt ** 2) * ((params.FT.mtimes(params.P * params.sus_mask * m)) - params.data))
    )
    grad_Tik = params.P ** 2 * params.Tik_mask ** 2 * m
    grad = 2 * grad_Res + params.TV_reg * grad_TV + 2 * params.Tik_reg * grad_Tik + params.TV_reg2 * grad_TV2
    return grad




