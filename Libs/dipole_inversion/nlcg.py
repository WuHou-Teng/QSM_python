import numpy as np
from tools.CallBackTools import CallBackTool


def nlcg(m0, params):
    """
    Phi(m) = ||W(Fu*m - y)||^2 + lamda1*|TV*m|_1
    m: susceptibility
    W: weighting matrix derived from magnitude intensities
    Fu: F_{-1}*D*F forward calculates the field from susceptibility
    y: measured field to be fitted (inversion)
    lambda1: TV regularization parameter
    TV: total variation operation

    ||...||^2: L2 norm
    |...|_1: L1 norm

    note the TV term can also be L2 norm if set p=2,
    then the term would be changed to ||TV*m||^2

    :param m0:
    :param params:
    :return:
        [m,RES,TVterm] : susceptibility distribution after dipole inversion
    """

    m = m0

    # line search parameters
    maxlsiter = params.lineSearchItnlim
    gradToll  = params.gradToll
    alpha     = params.lineSearchAlpha
    beta      = params.lineSearchBeta
    t0        = params.lineSearchT0
    k         = 0

    # compute (-gradient): search direction
    g0 = wGradient(m, params)
    dm = -g0

    f = 0
    RES0 = 0
    count = 0

    # 设定一个监控
    callback = CallBackTool()
    print("运行到 nlcg 第47行，开始迭代。")
    callback.print_time()

    # iterations
    while k < params.Itnlim:

        # backtracking line-search
        t = t0
        # def objFunc(m, dm, t, params)
        [f0, RES0, TVterm0] = objFunc(m, dm, 0, params)
        [f1, RES, TVterm] = objFunc(m, dm, t, params)
        lsiter = 0
        callback.print_time_line(59)
        #     while (f1 > f0 + alpha*t*(g0(:)'*dm(:))) && (lsiter<maxlsiter)
        #         t = t* beta;
        #         [f1, RES, TVterm] = objFunc(m,dm,t,params);
        #         lsiter = lsiter + 1;
        #     end

        while (f1 > (f0 + alpha * t * (np.reshape(g0, (np.size(g0), 1))).T.dot(np.reshape(dm, (np.size(dm), 1))))
               and lsiter < maxlsiter):
            t = t * beta
            [f1, RES, TVterm] = objFunc(m, dm, t, params)
            lsiter = lsiter + 1
            callback.print_time_line(71)
        callback.print_time_line(72)
        # control the number of line searches by adapting the initial step search
        if lsiter > 2:
            t0 = t0 * beta

        if lsiter < 1:
            t0 = t0 / beta

        # updates
        m = m + t * dm
        dm0 = dm
        g1 = wGradient(m, params)
        # TODO eps 存疑
        # bk = g1(:)'*g1(:)/(g0(:)' * g0(:)+eps)
        bk = (
                np.reshape(g1, (np.size(g1), 1)).T.dot(np.reshape(g1, (np.size(g1), 1)))
                / (np.reshape(g1, (np.size(g0), 1)).T.dot(np.reshape(g1, (np.size(g0), 1))) + np.finfo(float).eps)
        )
        g0 = g1
        dm = -g1 + bk * dm
        k = k + 1
        callback.print_time_line(93)
        # outputs for debugging purpose
        # fprintf('%d , relative residual: %f\n',...
        #         k, abs(RES-RES0)/RES);

        # if (abs(RES-RES0)/RES <= gradToll);
        #     count = count + 1;
        # else
        #     count = 0;
        # end
        # fprintf('  # d , relative changes: %f\n', k, norm(t * dm(:)) / norm(m(:)))
        print(f"{k} , relative residual: "
              f"{np.linalg.norm(t * np.reshape(dm, (np.size(dm), 1))) / np.linalg.norm(np.reshape(m, (np.size(m), 1)))}")
        callback.print_time_iter(k)
        callback.print_predict_time(k, params.Itnlim)
        callback.print_time_line(108)
        # if (norm(t * dm(:)) / norm(m(:)) <= gradToll):
        if np.linalg.norm(t * dm) / np.linalg.norm(m) <= gradToll:
            count = count + 1
        else:
            count = 0

        if count == 10:
            break

        f = f1
        RES0 = RES
    callback.print_time_line(120)
    return m, RES, TVterm


def objFunc(m, dm, t, params):
    callback = CallBackTool()
    callback.print_time_function_start("objFunc")
    p = params.pNorm
    w1 = m + t * dm

    w2 = params.TV.mtimes(w1)
    # TV = (w2.*conj(w2)+params.l1Smooth).^(p/2)
    TV = (w2 * np.conjugate(w2) + params.l1Smooth) ** (p / 2)
    # TVterm = sum(params.TVWeight(:).*TV(:))
    TVterm = np.sum(params.TVWeight * TV)

    RES = params.FT.mtimes(w1) - params.data
    # RES = (params.wt(:).*RES(:))'*(params.wt(:).*RES(:))
    # matlab 中直接用 (:) 对数组进行遍历会直接将数组视作一维数组，而非矩阵，因此这里需要用 reshape 函数将原本的多维数组转换为1维的。
    RES = (np.reshape(params.wt, (np.size(params.wt), 1)) * np.reshape(RES, (np.size(RES), 1))).T.dot(
        np.reshape(params.wt, (np.size(params.wt), 1)) * np.reshape(RES, (np.size(RES), 1))
    )

    obj = RES + TVterm
    callback.print_time_function_end("objFunc")
    return obj, RES, TVterm


def wGradient(m, params):
    callback = CallBackTool()
    callback.print_time_function_start("wGradient")
    p = params.pNorm
    # params.TV 是cls_tv类，在matlab中，代码对mtimes进行了重载，而mtimes在matlab中是代表乘法的内置函数。
    # 因此，这里需要调用mtimes函数，而非直接打乘号
    # w1 = params.TV.adjoint * m
    w1 = params.TV.mtimes(m)

    # matlab 中对 ctranspose 也进行了重载，因此这里需要调用 ctranspose 函数，而非直接打转置号
    # gradTV = params.TV'*(p*w1.*(w1.*conj(w1)+params.l1Smooth).^(p/2-1))
    gradTV = params.TV.trans.mtimes(
        p * w1 * (w1 * np.conjugate(w1) + params.l1Smooth) ** (p / 2 - 1)
    )
    # gradRES = params.FT'*((params.wt.^2).*((params.FT*m)-params.data))
    gradRES = params.FT.trans.mtimes(
        ((params.wt ** 2) * ((params.FT.mtimes(m)) - params.data))
    )

    grad = 2 * gradRES + gradTV * params.TVWeight
    callback.print_time_function_end("wGradient")
    return grad
