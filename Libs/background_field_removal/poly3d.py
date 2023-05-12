import numpy as np


# TODO 尚未debug
def poly3d(lfs, mask, poly_order=2):

    # [np nv nv2] = size(lfs);
    npp, nv, nv2 = np.shape(lfs)
    # % polyfit
    # px = repmat((1:np)',[nv*nv2,1]);
    px = np.tile(np.arange(1, npp + 1), (nv * nv2, 1))
    px = np.reshape(px, (np.size(px), 1))

    # py = repmat((1:nv),[np,1]);
    py = np.tile(np.arange(1, nv + 1), (np, 1))
    # py = repmat(py(:),[nv2,1]);
    py = np.tile(np.reshape(py, (np.size(py), 1)), (nv2, 1))
    # pz = repmat((1:nv2),[np*nv,1]);
    pz = np.tile(np.arange(1, nv2 + 1), (np * nv, 1))
    # pz = pz(:);
    pz = np.reshape(pz, (np.size(pz), 1))

    # % fit only the non-zero region
    # px_nz = px(logical(mask(:)));
    px_nz = px[np.where(mask)]
    # py_nz = py(logical(mask(:)));
    py_nz = py[np.where(mask)]
    # pz_nz = pz(logical(mask(:)));
    pz_nz = pz[np.where(mask)]

    # if poly_order == 1
    # 	% first order polyfit
    # 	P = [px, py, pz, ones(length(px),1)];
    # 	P_nz = [px_nz, py_nz, pz_nz, ones(length(px_nz),1)]; % polynomials
    if poly_order == 1:
        # first order polyfit
        P = np.concatenate((px, py, pz, np.ones((np.size(px), 1))), axis=1)
        P_nz = np.concatenate((px_nz, py_nz, pz_nz, np.ones((np.size(px_nz), 1))), axis=1)

    # elseif poly_order == 2
    # 	% second order
    # 	P = [px.^2, py.^2, pz.^2, px.*py, px.*pz, py.*pz, px, py, pz, ones(length(px),1)]; % polynomials
    # 	P_nz = [px_nz.^2, py_nz.^2, pz_nz.^2, px_nz.*py_nz, px_nz.*pz_nz, py_nz.*pz_nz, px_nz, py_nz, pz_nz,
    # 	ones(length(px_nz),1)]; % polynomials
    elif poly_order == 2:
        # second order
        P = np.concatenate((px ** 2, py ** 2, pz ** 2,
                            px * py, px * pz, py * pz,
                            px, py, pz, np.ones((np.size(px), 1))), axis=1)
        P_nz = np.concatenate((px_nz ** 2, py_nz ** 2, pz_nz ** 2,
                               px_nz * py_nz, px_nz * pz_nz, py_nz * pz_nz,
                               px_nz, py_nz, pz_nz, np.ones((np.size(px_nz), 1))), axis=1)

    # elseif poly_order == 3
    # 	% third order
    # 	P = [px.^3, py.^3, pz.^3, px.*py.^2, px.*pz.^2, px.*py.*pz, py.*px.^2, py.*pz.^2,
    # 	     pz.*px.^2, pz.*py.^2, px.^2, py.^2, pz.^2, px.*py, px.*pz, py.*pz, px, py, pz,
    # 	     ones(length(px),1)]; % polynomials
    # 	P_nz = [px_nz.^3, py_nz.^3, pz_nz.^3, px_nz.*py_nz.^2, px_nz.*pz_nz.^2, px_nz.*py_nz.*pz_nz,
    # 	        py_nz.*px_nz.^2, py_nz.*pz_nz.^2, pz_nz.*px_nz.^2, pz_nz.*py_nz.^2, px_nz.^2, py_nz.^2,
    # 	        pz_nz.^2, px_nz.*py_nz, px_nz.*pz_nz, py_nz.*pz_nz, px_nz, py_nz, pz_nz,
    # 	        ones(length(px_nz),1)]; % polynomials
    elif poly_order == 3:
        # third order
        P = np.concatenate((px ** 3, py ** 3, pz ** 3,
                            px * py ** 2, px * pz ** 2, px * py * pz,
                            py * px ** 2, py * pz ** 2,
                            pz * px ** 2, pz * py ** 2,
                            px ** 2, py ** 2, pz ** 2,
                            px * py, px * pz, py * pz,
                            px, py, pz, np.ones((np.size(px), 1))), axis=1)
        P_nz = np.concatenate((px_nz ** 3, py_nz ** 3, pz_nz ** 3,
                               px_nz * py_nz ** 2, px_nz * pz_nz ** 2, px_nz * py_nz * pz_nz,
                               py_nz * px_nz ** 2, py_nz * pz_nz ** 2,
                               pz_nz * px_nz ** 2, pz_nz * py_nz ** 2,
                               px_nz ** 2, py_nz ** 2, pz_nz ** 2,
                               px_nz * py_nz, px_nz * pz_nz, py_nz * pz_nz,
                               px_nz, py_nz, pz_nz, np.ones((np.size(px_nz), 1))), axis=1)

    # elseif poly_order == 4
    # 	% third order
    # 	P = [px.^4, py.^4, pz.^4, px.^3.*py, px.^3.*pz, py.^3.*px, py.^3.*pz, pz.^3.*px, pz.^3.*py, px.^2.*py.^2,
    #        px.^2.*pz.^2, px.^2.*py.*pz, py.^2.*pz.^2, py.^2.*px.*pz, pz.^2.*px.*py, px.^3, py.^3, pz.^3, px.*py.^2,
    #        px.*pz.^2, px.*py.*pz, py.*px.^2, py.*pz.^2, pz.*px.^2, pz.*py.^2, px.^2, py.^2, pz.^2, px.*py, px.*pz,
    #        py.*pz, px, py, pz, ones(length(px),1)]; % polynomials
    # 	P_nz = [px_nz.^4, py_nz.^4, pz_nz.^4, px_nz.^3.*py_nz, px_nz.^3.*pz_nz, py_nz.^3.*px_nz, py_nz.^3.*pz_nz,
    #           pz_nz.^3.*px_nz, pz_nz.^3.*py_nz, px_nz.^2.*py_nz.^2, px_nz.^2.*pz_nz.^2, px_nz.^2.*py_nz.*pz_nz,
    #           py_nz.^2.*pz_nz.^2, py_nz.^2.*px_nz.*pz_nz, pz_nz.^2.*px_nz.*py_nz, px_nz.^3, py_nz.^3, pz_nz.^3,
    #           px_nz.*py_nz.^2, px_nz.*pz_nz.^2, px_nz.*py_nz.*pz_nz, py_nz.*px_nz.^2, py_nz.*pz_nz.^2,
    #           pz_nz.*px_nz.^2, pz_nz.*py_nz.^2, px_nz.^2, py_nz.^2, pz_nz.^2, px_nz.*py_nz, px_nz.*pz_nz,
    #           py_nz.*pz_nz, px_nz, py_nz, pz_nz, ones(length(px_nz),1)]; % polynomials
    elif poly_order == 4:
        # forth order
        P = np.concatenate((px ** 4, py ** 4, pz ** 4,
                            px ** 3 * py, px ** 3 * pz,
                            py ** 3 * px, py ** 3 * pz,
                            pz ** 3 * px, pz ** 3 * py,
                            px ** 2 * py ** 2, px ** 2 * pz ** 2, px ** 2 * py * pz,
                            py ** 2 * pz ** 2, py ** 2 * px * pz,
                            pz ** 2 * px * py,
                            px ** 3, py ** 3, pz ** 3,
                            px * py ** 2, px * pz ** 2, px * py * pz,
                            py * px ** 2, py * pz ** 2,
                            pz * px ** 2, pz * py ** 2,
                            px ** 2, py ** 2, pz ** 2,
                            px * py, px * pz, py * pz,
                            px, py, pz, np.ones((np.size(px), 1))), axis=1)
        P_nz = np.concatenate((px_nz ** 4, py_nz ** 4, pz_nz ** 4,
                               px_nz ** 3 * py_nz, px_nz ** 3 * pz_nz,
                               py_nz ** 3 * px_nz, py_nz ** 3 * pz_nz,
                               pz_nz ** 3 * px_nz, pz_nz ** 3 * py_nz,
                               px_nz ** 2 * py_nz ** 2,
                               px_nz ** 2 * pz_nz ** 2, px_nz ** 2 * py_nz * pz_nz,
                               py_nz ** 2 * pz_nz ** 2, py_nz ** 2 * px_nz * pz_nz,
                               pz_nz ** 2 * px_nz * py_nz,
                               px_nz ** 3, py_nz ** 3, pz_nz ** 3,
                               px_nz * py_nz ** 2, px_nz * pz_nz ** 2, px_nz * py_nz * pz_nz,
                               py_nz * px_nz ** 2, py_nz * pz_nz ** 2,
                               pz_nz * px_nz ** 2, pz_nz * py_nz ** 2,
                               px_nz ** 2, py_nz ** 2, pz_nz ** 2,
                               px_nz * py_nz, px_nz * pz_nz, py_nz * pz_nz,
                               px_nz, py_nz, pz_nz, np.ones((np.size(px_nz), 1))), axis=1)
    # else
    # 	error('cannot do higher than 3rd order');
    else:
        raise ValueError('cannot do higher than 4th order')
    # end

    # I = lfs(logical(mask)); % measurements of non-zero region
    I = lfs[np.where(mask)]
    # I = I(:);
    I = np.reshape(I, (np.size(I), 1))

    # % coeff = P_nz\I; % polynomial coefficients
    # 这行 Matlab 代码是对一个矩阵方程组进行求解，其中 P_nz 是一个 nxm 的矩阵，I 是一个 nx1 的向量。
    # 求解的过程是利用最小二乘法来拟合一个线性模型，即 I = P_nz · coeff，其中 coeff 是一个 mx1 的向量，表示拟合出的系数。
    # coeff = (P_nz'*P_nz)\(P_nz'*I);
    coeff = np.linalg.lstsq(P_nz, I, rcond=None)[0]
    # polyfit = P*coeff;
    polyfit = np.dot(P, coeff)
    # polyfit = reshape(polyfit,[np,nv,nv2]);
    polyfit = np.reshape(polyfit, (np, nv, nv2))

    # polyfit(isnan(polyfit)) = 0;
    polyfit[np.isnan(polyfit)] = 0
    # polyfit(isinf(polyfit)) = 0;
    polyfit[np.isinf(polyfit)] = 0

    return polyfit
