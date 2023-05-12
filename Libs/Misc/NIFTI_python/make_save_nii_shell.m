function make_save_nii_shell(data_name, vox, filename)
% MAKE_SAVE_NII 对makenii 和save nii的二次封装
% 该方法显然是不科学，缺乏可发展性的。
%   此处显示详细说明

    addpath(genpath("C:\Wuhou\homework\ELEC_Thesis\QSM_python\Libs\Misc\NIFTI"))
    fprintf("完成路径添加\n");
    ph_diff_cmb = load(data_name + "_temp.mat", data_name);
    fprintf("完成矩阵读取\n");
    nii = make_nii(angle(ph_diff_cmb.(data_name)),vox);
    fprintf("完成make_nii");
    save_nii(nii, filename);
    fprintf("完成保存nii");
end

