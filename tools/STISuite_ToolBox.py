import subprocess
import os
import matlab.engine
from scipy.io import loadmat, savemat
import numpy as np
import nibabel as nib


def QSM_iLSQR_eng(lfs_resharp, mask_resharp, z_prjs, vox, niter, TE, B0):
    """
    返回经过处理后保存好的文件的路径。
    :param lfs_resharp:
    :param mask_resharp:
    :param z_prjs:
    :param vox:
    :param niter:
    :param TE:
    :param B0:
    :return:
    """
    # 首先，保存文件到临时文件夹
    # 首先获取绝对路径
    root = os.path.dirname(os.path.abspath(__file__)).split("QSM_python")[0]
    current_dir = os.path.join(root, "QSM_python", "tools", "STISuite_V3.0", "STISuiteTools_pack_for_engine")
    # 启动engine
    eng = matlab.engine.start_matlab()
    eng.addpath(current_dir, nargout=0)
    eng.addpath(current_dir + "../STISuite_V3.0/Core_Functions_P", nargout=0)
    # 创建临时文件夹Temp
    temp_dir = os.path.join(current_dir, "Temp")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    # 将lfs_resharp 和 mask_resharp 保存到临时文件夹
    lfs_resharp_path = os.path.join(temp_dir, "lfs_resharp.mat")
    mask_resharp_path = os.path.join(temp_dir, "mask_resharp.mat")
    savemat(lfs_resharp_path, {"lfs_resharp": lfs_resharp})
    savemat(mask_resharp_path, {"mask_resharp": mask_resharp})

    # 准备matlab调用指令
    cmd = f"QSM_iLSQR_eng('{lfs_resharp_path}','{mask_resharp_path}',{z_prjs},{vox},{niter},{TE},{B0})"
    # 调用
    eng.eval(cmd, nargout=0)
    eng.quit()
    return os.path.join(temp_dir, "QSM_iLSQR.mat")




