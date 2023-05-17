import numpy as np
from scipy import fft
from Libs.dipole_inversion.cls_tv import Cls_TV
from Libs.dipole_inversion.cls_dipconv import Cls_Dipconv
from tools.StructureCLass import Param
from Libs.dipole_inversion.nlcg_tik import nlcg_tik


def tikhonov_qsm(tfs, Res_wt, sus_mask, TV_mask, Tik_mask, air_mask, TV_reg, Tik_reg, TV_reg2,
                 vox, P=None, z_prjs=None, Itnlim=None):
    # 参数检查和设置默认值
    if P is None:
        P = 1  # 无预处理
    if z_prjs is None:
        z_prjs = [0, 0, 1]  # PURE轴向切片
    if Itnlim is None:
        Itnlim = 500

    # 创建K空间滤波器核函数D
    Nx, Ny, Nz = tfs.shape
    FOV = vox * np.array([Nx, Ny, Nz])
    FOVx, FOVy, FOVz = FOV

    x = np.arange(-Nx / 2, Nx / 2)
    y = np.arange(-Ny / 2, Ny / 2)
    z = np.arange(-Nz / 2, Nz / 2)

    kx, ky, kz = np.meshgrid(x / FOVx, y / FOVy, z / FOVz)
    D = 1 / 3 - (kx * z_prjs[0] + ky * z_prjs[1] + kz * z_prjs[2]) ** 2 / (kx ** 2 + ky ** 2 + kz ** 2)
    D[Nx // 2, Ny // 2, Nz // 2] = 0
    D = np.fft.fftshift(D)

    # params.FT               = cls_dipconv([Nx, Ny, Nz], D); % class for dipole kernel convolution
    # params.TV               = cls_tv; 						% class for TV operation
    #
    # params.Itnlim           = Itnlim; 						% interations numbers (adjust accordingly!)
    # params.gradToll         = 1e-4; 						% step size tolerance stopping criterea
    # params.l1Smooth         = eps; 							% 1e-15; smoothing parameter of L1 norm
    # params.pNorm            = 1; 							% type of norm to use (i.e. L1 L2 etc)
    # params.lineSearchItnlim = 100;
    # params.lineSearchAlpha  = 0.01;
    # params.lineSearchBeta   = 0.6;
    # params.lineSearchT0     = 1 ; 							% step size to start with
    #
    # params.Tik_reg          = Tik_reg;
    # params.TV_reg           = TV_reg;
    # params.Tik_mask         = Tik_mask;
    # params.TV_mask          = TV_mask;
    # params.sus_mask         = sus_mask;
    # params.Res_wt           = Res_wt;
    # params.data             = tfs;
    # params.P                = P;
    params = Param()
    params.FT = Cls_Dipconv([Nx, Ny, Nz], D)
    params.TV = Cls_TV()
    params.Itnlim = Itnlim
    params.gradToll = 1e-4
    params.l1Smooth = np.finfo(float).eps   # 1e-15
    params.pNorm = 1
    params.lineSearchItnlim = 100
    params.lineSearchAlpha = 0.01
    params.lineSearchBeta = 0.6
    params.lineSearchT0 = 1

    params.Tik_reg = Tik_reg
    params.TV_reg = TV_reg
    params.Tik_mask = Tik_mask
    params.TV_mask = TV_mask
    params.sus_mask = sus_mask
    params.Res_wt = Res_wt
    params.data = tfs
    params.P = P

    # 非线性共轭梯度法
    chi, Res_term, TV_term, Tik_term = nlcg_tik(np.zeros([Nx, Ny, Nz]), params)

    # % LSQR method
    # % argmin ||Res_wt * (F_{-1} * D * F * sus_mask * chi - tfs)|| + Tik_reg * ||Tik_mask * chi||
    #
    # % b = P.*sus_mask.*ifftn(D.*fftn(Res_wt.*Res_wt.*tfs));
    # % b = b(:);
    # % imsize = size(tfs);
    # % m = lsqr(@Afun, b, 1e-6, Itnlim);
    # % chi = real(reshape(m,imsize));
    #
    # % function y = Afun(x,transp_flag)
    # % if strcmp(transp_flag,'transp')
    # % % y = H'*(H*x) + tik_reg*x;
    # % x = reshape(x,imsize);
    # % y = P.*sus_mask.*ifftn(D.*fftn(Res_wt.*Res_wt.*ifftn(D.*fftn(sus_mask.*P.*x)))) + Tik_reg*P.*Tik_mask.*Tik_mask.*P.*x;
    #
    # % y = y(:);
    #
    # % else
    # % % y = H'*(H*x) + tik_reg*x;
    # % x = reshape(x,imsize);
    # % y = P.*sus_mask.*ifftn(D.*fftn(Res_wt.*Res_wt.*ifftn(D.*fftn(sus_mask.*P.*x)))) + Tik_reg*P.*Tik_mask.*Tik_mask.*P.*x;
    #
    # %     y = y(:);
    # %     end
    # % end

    # 如果想保留偶极矩形拟合结果
    # 不要对其进行遮罩，而是使用以下代码：
    chi = np.real(params.P * chi)
    # 否则对其进行遮罩
    return chi, Res_term, TV_term, Tik_term
