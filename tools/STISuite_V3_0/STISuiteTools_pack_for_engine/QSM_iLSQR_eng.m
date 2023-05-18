function chi_iLSQR_eng = QSM_iLSQR_eng(lfs_resharp_file, mask_resharp_file, z_prjs, vox, niter, TE, B0)
    % 从 .mat 文件中加载 lfs_resharp 和 mask_resharp
    lfs_resharp_data = load(lfs_resharp_file);
    mask_resharp_data = load(mask_resharp_file);

    lfs_resharp = lfs_resharp_data.lfs_resharp;
    mask_resharp = mask_resharp_data.mask_resharp;
    % 这里路径是死的，但是暂时没时间改了。
    % addpath(genpath("/home/wuhou/文档/work/ELEC_Thesis/QSM_python/tools/STISuite_V3_0/STISuite_V3_0/Core_Functions_P"))

    % 调用 QSM_iLSQR 函数
    chi_iLSQR = QSM_iLSQR(lfs_resharp , mask_resharp, 'H', z_prjs, 'voxelsize', vox, 'niter', niter, 'TE', TE, 'B0', B0);

    % 返回结果
    chi_iLSQR_eng = chi_iLSQR;

    % 保存结果
    save('/home/wuhou/文档/work/ELEC_Thesis/QSM_python/tools/STISuite_V3_0/STISuiteTools_pack_for_engine/Temp/chi_iLSQR.mat', 'chi_iLSQR');
end
