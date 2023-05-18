
function chi_iLSQR_eng = QSM_iLSQR_eng(lfs_resharp_file, mask_resharp_file, dicom_info, z_prjs, vox, niter, TE, B0)
    % 从 .mat 文件中加载 lfs_resharp 和 mask_resharp
    lfs_resharp_data = load(lfs_resharp_file);
    mask_resharp_data = load(mask_resharp_file);

    lfs_resharp = lfs_resharp_data.lfs_resharp;
    mask_resharp = mask_resharp_data.mask_resharp;

    % 调用 QSM_iLSQR 函数
    chi_iLSQR = QSM_iLSQR(lfs_resharp * (2.675e8 * dicom_info.MagneticFieldStrength) / 1e6, mask_resharp, 'H', z_prjs, 'voxelsize', vox, 'niter', 50, 'TE', 1000, 'B0', dicom_info.MagneticFieldStrength);

    % 返回结果
    chi_iLSQR_eng = chi_iLSQR;

    % 保存结果
    save('chi_iLSQR.mat', 'chi_iLSQR');
end
