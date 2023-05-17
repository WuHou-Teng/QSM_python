function make_save_nii_eng(data_name, vox, filename)
% MAKE_SAVE_NII 对makenii 和save nii的二次封装
% 该方法显然是不科学，缺乏可发展性的。
%   此处显示详细说明


    % addpath(genpath("C:\Wuhou\homework\ELEC_Thesis\QSM_python\Libs\Misc\NIFTI"))
    % fprintf(fid, "完成路径添加\n");
    % 将文件名称与完整目录分离
    [data_path,name,ext] = fileparts(data_name);
    data = load(data_name + "_temp.mat", name);
    % fprintf(fid, "完成矩阵读取\n");
    nii = make_nii(data.(name),vox);
    % fprintf(fid, "完成make_nii\n");
    save_nii(nii, filename);
    % fprintf(fid, "完成保存nii\n");

    % fclose(fid);
end
