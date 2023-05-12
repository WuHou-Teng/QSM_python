import subprocess
import os
import matlab.engine
from scipy.io import loadmat, savemat
import numpy as np


def make_save_nii_shell(data, data_name, vox, filename):
    """
    生成 NIFTI 文件
    :param data: 数据
    :param data_name: 数据名称
    :param vox: 体素大小
    :param filename: 文件名称
    :return:
    """
    # 首先将矩阵保存下来。
    data_name = os.path.join(os.getcwd() + "\\..\\..\\Temp_files", data_name)
    savemat(data_name + "_temp.mat", {data_name: data})
    filename = os.path.join(os.getcwd() + "\\..\\..\\Temp_files", filename)

    # 定义 Matlab 命令
    # cmd = "matlab -r 'make_save_nii(angle(phase_diff_cmb),vox,filename); exit'"
    cmd = f"matlab -r \"make_save_nii_shell('{data_name}',{vox},'{filename}'); exit\""

    print(cmd)

    # 运行 Matlab 命令行
    subprocess.run(cmd, shell=True)


def make_save_nii_engine(data, data_name, vox, filename, del_temp=False):
    """
    通过matlab engine来调用，无需新窗口，效果尚可。
    vox的长度 是根据data的形状来确定的，如果是三维，则是[1,1,1], 如果是四维，则是[1,1,1,1]
    :param del_temp:
    :param data:
    :param data_name:
    :param vox:
    :param filename:
    :return:
    """
    # 首先获取绝对路径
    root = os.path.dirname(os.path.abspath(__file__)).split("Libs")[0]
    current_dir = os.path.join(root, "Libs", "Misc", "NIFTI_python")
    eng = matlab.engine.start_matlab()
    eng.addpath(current_dir, nargout=0)
    eng.addpath(current_dir + "\\..\\NIFTI", nargout=0)

    # 首先将矩阵保存下来。()
    data_path = os.path.join(current_dir + "\\..\\..\\Temp_files", data_name)
    savemat(data_path + "_temp.mat", {data_name: data})
    filename = os.path.join(current_dir + "\\..\\..\\Temp_files", filename)

    # 运行相应的指令
    # eng.make_save_nii_eng(data_name, vox, filename, nargout=0)
    # eng.make_save_nii_eng(data_name, vox, filename, nargout=0)
    # if angle:
    #     eng.eval(f"make_save_nii_eng_angle('{data_name}',{vox},'{filename}');", nargout=0)
    # else:
    #     eng.eval(f"make_save_nii_eng('{data_name}',{vox},'{filename}');", nargout=0)
    eng.eval(f"make_save_nii_eng('{data_path}',{vox},'{filename}');", nargout=0)
    eng.quit()
    # 删除临时文件
    if del_temp:
        os.remove(data_path + "_temp.mat")


def make_save_nii_engine2(data, vox, filename):
    """
    直接传递矩阵，而不是通过文件的方式，事实证明速度太慢，不采用。
    :param data:
    :param data_name:
    :param vox:
    :param filename:
    :return:
    """
    eng = matlab.engine.start_matlab()
    eng.addpath(r"Libs\Misc\NIFTI_python", nargout=0)
    eng.addpath(r"Libs\Misc\NIFTI", nargout=0)

    # 调整数据类型
    mat_data = eng.double(data.tolist())
    # 运行相应的指令
    # eng.make_save_nii_eng(data_name, vox, filename, nargout=0)
    eng.make_save_nii_eng2(mat_data, vox, filename, nargout=0)
    # eng.eval(f"make_save_nii_eng2('{data}',{vox},'{filename}');", nargout=0)
    eng.quit()


if __name__ == "__main__":
    lfs = loadmat("./lfs.mat")  # scipy.io.loadmat
    # make_save_nii_engine(lfs["lfs"], "lfs", [1, 1, 1], "lfs.nii")
    make_save_nii_engine(lfs["lfs"], "lfs", [1, 1, 1], "lfs.nii")

