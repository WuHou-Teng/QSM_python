import numpy as np


def padarray(ndarray, pad_shape, value=0) -> np.array:
    """
    对 matlab 函数 padarray的重写。用来方便重构代码。
    :param ndarray: 要pad的数组
    :param pad_shape: 要往数组 pad的形状, 可以是list, 也可以是ndarray
    :param value:
    :return:
    """
    # 首先获取array的shape
    dimension = ndarray.shape
    pad = []
    for i in range(len(dimension)):
        # 如果是list，肯定是一维的，直接用index调用即可
        if type(pad_shape) is list:
            pad.append([int(pad_shape[i]), int(pad_shape[i])])
        else:
            # 不是list，那就必须是 ndarray，要判断是一维还是二维
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


def ma_range(start, stop, step=1):
    """
    对 np.arange的重写，使函数形式更接近 matlab。
        将stop的取值改为 stop+step
    :param start: 开始值
    :param step: 步长
    :param stop: 结束值
    :return:
    """
    return np.arange(start, stop+step, step)


def nd_grid(*xi, copy=True, sparse=False, indexing='ij'):
    """
    对 np.meshgrid的重写，使函数形式更接近 matlab。
    :param xi:
    :param copy:
    :param sparse:
    :param indexing: 'xy' or 'ij',
        In the 2-D case with inputs of length M and N, the outputs are of shape
            (N, M) for 'xy' indexing and (M, N) for 'ij' indexing.
        In the 3-D case with inputs of length M, N and P,
            outputs are of shape (N, M, P) for 'xy' indexing
            and (M, N, P) for 'ij' indexing.
    :return:
    """

    out = np.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)
    out = list(out)
    # 将out的第一个元素放到最后
    out.append(out.pop(0))
    # 将out的最后一个元素放到最前
    # out.insert(0, out.pop())
    return out


def nd_grid_original(*xi, copy=True, sparse=False, indexing='ij'):
    """
    对 np.meshgrid的重写，使函数形式更接近 matlab。
    :param xi:
    :param copy:
    :param sparse:
    :param indexing: 'xy' or 'ij',
        In the 2-D case with inputs of length M and N, the outputs are of shape
            (N, M) for 'xy' indexing and (M, N) for 'ij' indexing.
        In the 3-D case with inputs of length M, N and P,
            outputs are of shape (N, M, P) for 'xy' indexing
            and (M, N, P) for 'ij' indexing.
    :return:
    """

    out = np.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)
    out = list(out)
    # 将out的第一个元素放到最后
    # out.append(out.pop(0))
    # 将out的最后一个元素放到最前
    # out.insert(0, out.pop())
    return out