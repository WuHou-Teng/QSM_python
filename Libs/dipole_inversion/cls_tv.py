import numpy as np


class Cls_TV(object):
    # CLS_TV Class for total variation

    def __init__(self, adjoint=0):
        self.adjoint = adjoint

    # no constructor function needed, since it takes no inputs (properties)

    def ctranspose(self, obj):
        obj.adjoint = obj.adjoint ^ 1
        return obj

    @property
    def trans(self):
        return Cls_TV(self.adjoint ^ 1)

    def mtimes(self, b):
        if self.adjoint:
            product = invD(b)
        else:
            product = D(b)

        return product


def invD(y):
    # res = adjDx(y(:,:,:,1)) + adjDy(y(:,:,:,2)) + adjDz(y(:,:,:,3));
    res = adjDx(y[:, :, :, 0]) + adjDy(y[:, :, :, 1]) + adjDz(y[:, :, :, 2])
    # 这里拓展了维度，所以需要用np.stack函数
    # res = np.stack((adjDx(y), adjDy(y), adjDz(y)), axis=-1)
    return res


def D(x):
    # Dx = x([2:end,end],:,:) - x
    Dx = x[[i for i in range(1, np.shape(x)[0])] + [-1], :, :] - x
    # Dy = x(:,[2:end,end],:) - x
    Dy = x[:, [i for i in range(1, np.shape(x)[1])] + [-1], :] - x
    # Dz = x(:,:,[2:end,end]) - x
    Dz = x[:, :, [i for i in range(1, np.shape(x)[2])] + [-1]] - x
    # res = cat(4,Dx,Dy,Dz)
    # 要新创建一个轴，需要用stack函数连接，并将axis设定为-1
    res = np.stack((Dx, Dy, Dz), axis=-1)
    # res = np.concatenate((Dx, Dy, Dz), axis=3)
    return res


def adjDx(x):
    # res = x([1,1:end-1],:,:) - x
    res = x[[0] + [i for i in range(0, np.shape(x)[0] - 1)], :, :] - x
    # res(1,:,:) = -x(1,:,:)
    res[0, :, :] = -x[0, :, :]
    # res(end,:,:) = x(end-1,:,:)
    res[-1, :, :] = x[-2, :, :]
    return res


def adjDy(x):
    # res = x(:,[1,1:end-1],:) - x
    res = x[:, [0] + [i for i in range(0, np.shape(x)[1] - 1)], :] - x
    # res(:,1,:) = -x(:,1,:)
    res[:, 0, :] = -x[:, 0, :]
    # res(:,end,:) = x(:,end-1,:)
    res[:, -1, :] = x[:, -2, :]
    return res


def adjDz(x):
    # res = x(:,:,[1,0:end-1]) - x
    res = x[:, :, [0] + [i for i in range(0, np.shape(x)[2] - 1)]] - x
    # res(:,:,1) = -x(:,:,1)
    res[:, :, 0] = -x[:, :, 0]
    # res(:,:,end) = x(:,:,end-1)
    res[:, :, -1] = x[:, :, -2]
    return res
