import numpy as np
import time

from scipy.io import loadmat, savemat
from scipy.signal import convolve
# 稀疏矩阵求解
from scipy.sparse.linalg import cgs, LinearOperator

# 密集矩阵求解
from scipy.linalg import solve


def resharp(tfs, mask, vox=None, ker_rad=3, tik_reg=1e-4, iter_num=200):
    """
    [LSF,MASK_ERO,RES_TERM,REG_TERM] = RESHARP(TFS,MASK,VOX,KER_RAD,TIK_REG)
    Method is described in the paper:
    Sun, H. and Wilman, A. H. (2013),
    Background field removal using spherical mean value filtering and Tikhonov regularization.
    Magn Reson Med. doi: 10.1002/mrm.24765

    :param tfs: input total field shift
    :param mask: binary mask defining the brain ROI
    :param vox: voxel size (mm), e.g. [1,1,1] for isotropic
    :param ker_rad: radius of convolution kernel (mm), e.g. 3
    :param tik_reg: Tikhonov regularization parameter, e.g. 1e-4
    :param iter_num: maximum number of CGS times of iteration, e.g. 200
    :return:
        LFS         : local field shift after background removal
        MASK_ERO    : eroded mask after convolution
        RES_TERM    : norm of data fidelity term
        REG_TERM    : norm of regularization term
    """

    # FIXME 设定一个监控
    callback = CallBackTool()
    callback.print_time(36)

    if vox is None:
        vox = np.array([1, 1, 1])

    # make spherical/ellipsoidal convolution kernel (ker)
    rx = np.round(ker_rad / vox[0])
    ry = np.round(ker_rad / vox[1])
    rz = np.round(ker_rad / vox[2])
    rx = max(rx, 2)
    ry = max(ry, 2)
    rz = max(rz, 2)
    # rz = ceil(ker_rad/vox(3));
    # TODO 关于 ndgrid, 三维的矩阵需要将输出x移动到最后才能保证和matlab相同。
    # [X, Y, Z] = ndgrid(-rx:rx, -ry: ry, -rz: rz)
    Y, Z, X = np.meshgrid(np.arange(-rx, rx + 1),
                          np.arange(-ry, ry + 1),
                          np.arange(-rz, rz + 1))
    # 矩阵计算
    # 矩阵乘法 np.dot()
    # 矩阵幂运算 np.power()
    # TODO 数组条件性创建
    h = (np.power(X, 2) / rx ** 2 +
         np.power(Y, 2) / ry ** 2 +
         np.power(Z, 2) / rz ** 2)
    for index in np.ndindex(h.shape):
        if h[index] > 1:
            h[index] = 0
    # 这里的sum将矩阵中所有的数值相加的效果。必须得有np.sum
    ker = h / np.sum(h[:])

    # pad zeros around to avoid errors in trans between linear conv and FT multiplication
    # np.pad 第二部分的参数，每一个元组, 例如(3,3)，都代表了数组在一个维度的最开始和最末尾需要pad的层数。
    # 简而言之((3, 3), (3, 3), (3, 3))表示在数组的 (上，下), (左，右), (前，后) 各添加三层。
    # tfs = np.pad(tfs, ((3, 3), (3, 3), (3, 3)), mode='constant', constant_values=0)
    # mask = np.pad(mask, ((3, 3), (3, 3), (3, 3)), mode='constant', constant_values=0)
    # TODO 重构的代码已经写在下面了。

    # circular_shift, linear conv to Fourier multiplication
    csh = np.array([[rx, ry, rz]])  # circular_shift

    tfs = padarray(tfs, csh)
    mask = padarray(mask, csh)

    imsize = tfs.shape

    mask_ero = np.zeros(imsize)
    mask_tmp = convolve(mask, ker, "same")  # scipy.signal.convolve

    # mask_ero(mask_tmp > 1-1/sum(h(:))) = 1; % no error points tolerence
    # mask_ero(mask_tmp > 0.999999) = 1  # no error points tolerence
    for index in np.ndindex(mask_ero.shape):
        try:
            if mask_tmp[index] > 0.999999:
                mask_ero[index] = 1
        except:  # 可能出现体积对不上的问题，直接跳过
            continue

    # prepare convolution kernel: delta-ker
    dker = -ker
    dker[int(rx + 1 - 1), int(ry + 1 - 1), int(rz + 1 - 1)] = 1 - ker[int(rx + 1 - 1), int(ry + 1 - 1), int(rz + 1 - 1)]
    # TODO fftn 的变形
    DKER = np.fft.fftn(dker, s=imsize)  # dker in Fourier domain

    # RESHARP with Tikhonov regularization:
    # argmin ||MSfCFx - MSfCFy||2 + lambda||x||2
    # x: local field
    # y: total field
    # M: binary mask
    # S: circular shift
    # F: forward FFT, f: inverse FFT (ifft)
    # C: deconvolution kernel
    # lambda: tikhonov regularization parameter
    # ||...||2: sum of square
    #
    # create 'MSfCF' as an object 'H', then simplified as:
    # argmin ||Hx - Hy||2 + lambda||x||2
    # To solve it, derivative equals 0:
    # (H'H + lambda)x = H'Hy
    # Model as Ax = b, solve with cgs

    # H = cls_smvconv(imsize,DKER,csh,mask_ero);
    b = np.fft.ifftn(
        np.conj(DKER) * np.fft.fftn(
            circshift(
                mask_ero * circshift(np.fft.ifftn(DKER * np.fft.fftn(tfs)), -csh),
                csh
            )
        )
    )
    # 整个展平一个ndarray，在matlab中，b(:)是展为(n,1)，而在python中用b.flatten() 获得的是(n,) 。所以需要用reshape
    b = b.reshape(b.size, 1)

    # nested function
    def Afun(x):
        # y = H'*(H*x) + tik_reg*x;
        x = x.reshape(imsize)
        y = np.fft.ifftn(np.conj(DKER) * np.fft.fftn(
            circshift(mask_ero * circshift(np.fft.ifftn(DKER * np.fft.fftn(x)), -csh), csh)
        )) + tik_reg * x
        return y.reshape(y.size, 1)

    # TODO csg 迭代。
    A = LinearOperator((b.size, b.size), Afun)
    tol = 1e-6
    callback.print_time(141)
    m, info = cgs(A, b, tol=tol, maxiter=iter_num, callback=callback.csg_callback_func)
    callback.print_time(143)
    if info > 1e-6:
        print(f"cgs 在迭代 {iter_num} 停止，而没有收敛到所需容差 {tol}，"
              f"这是因为已达到最大迭代数。迭代返回的 (数目 {iter_num}) 的相对残差为 {info}。")
    else:
        print(f"已达到收敛所需容差{info}")

    lfs = np.real(m.reshape(imsize)) * mask_ero
    res_term = mask_ero * circshift(
        np.fft.ifftn(DKER * np.fft.fftn(tfs - m.reshape(imsize))), -csh
    )
    # TODO norm 替换
    res_term = np.linalg.norm(res_term.reshape(res_term.size, 1))
    reg_term = np.linalg.norm(m)

    # remove the padding for outputs
    # end 替换为空
    lfs = lfs[int(rx):int(-rx),
          int(ry):int(-ry),
          int(rz):int(-rz)]
    mask_ero = mask_ero[int(rx):int(-rx),
               int(ry):int(-ry),
               int(rz):int(-rz)]
    print("resharp 函数结束。")
    # img = nibabel.Nifti1Image(lfs, 0)
    # nibabel.save(img, "lfs")
    savemat("lfs.mat", {"lfs": lfs})
    return [lfs, mask_ero, res_term, reg_term]


def padarray(ndarray, pad_shape, value=0) -> np.array:
    """
    对 matlab 函数 padarray的重写。用来方便重构代码。
    :param ndarray: 要pad的数组
    :param pad_shape: 要往数组
    :param value:
    :return:
    """
    # 首先获取array的shape
    dimension = ndarray.shape
    pad = []
    for i in range(len(dimension)):
        if type(pad_shape) is list:
            pad.append([int(pad_shape[i]), int(pad_shape[i])])
        else:
            # 不是list，那就必须是 ndarray
            if len(pad_shape.shape) == 1:
                pad.append([int(pad_shape[i]), int(pad_shape[i])])
            elif len(pad_shape.shape) == 2:
                pad.append([int(pad_shape[0, i]), int(pad_shape[0, i])])
    return np.pad(ndarray, pad, mode='constant', constant_values=value)


def circshift(ndarray, shift, axis=None):
    if type(shift) is not list and tuple(shift):
        # shift 可能是个numpy数组
        shift_list = []
        try:
            for index in np.ndindex(shift.shape):
                shift_list.append(int(shift[index]))
        except:
            print("未知的shift")
    else:
        shift_list = tuple(shift)
    return np.roll(ndarray, shift_list, axis)


# 定义一个 callback 函数，用于输出迭代信息
class CallBackTool(object):
    def __init__(self):
        self.current_time = time.process_time()
        self.iter_num = 1

    def csg_callback_func(self, x):
        print(f"迭代次数:{self.iter_num}, 剩余残差为: {x}")
        self.iter_num += 1

    def print_time(self, line):
        main_time = time.process_time() - self.current_time
        self.current_time = time.process_time()
        print(f"程序总耗时:{main_time}秒, 运行到程序第{line}行")



if __name__ == "__main__":
    tfs_and_mask = loadmat("./tfs_and_mask.mat")  # scipy.io.loadmat
    # TODO 这里得在外面套一个array转换
    tfs = np.array(tfs_and_mask.get("tfs"))
    mask = np.array(tfs_and_mask.get("mask"))
    resharp(tfs, mask, iter_num=200)
