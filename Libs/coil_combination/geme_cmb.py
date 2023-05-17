import subprocess

import numpy as np
import nibabel as nib
import os

from scipy import ndimage

from Libs.background_field_removal.poly3d import poly3d
from Libs.background_field_removal.poly3d_nonlinear import poly3d_nonlinear


# function [ph_cmb,mag_cmb,coil_sens] = geme_cmb(img, vox, te, mask, smooth_method, parpool_flag)
def geme_cmb(img, vox, te, mask=None, smooth_method='gaussian', parpoll_flag=0):
    """
    Gradient-echo multi-echo combination (for phase).
    PH_CMB = GEME_CMB(IMG,VOX, TE,SMOOTH_METHOD) combines phase from multiple receivers
    :param img:             raw complex images from multiple receivers, 5D: [3D_image, echoes, receiver channels]
    :param vox:             echo times
    :param te:              brain mask
    :param mask: spatial    resolution/voxel size, e.g. [1 1 1] for isotropic
    :param smooth_method:   phase after combination
    :param parpoll_flag:    magnitude after combination
    :return:
    """

    # if ~ exist('mask','var') || isempty(mask)
    #     mask = ones(size(img,[1,2,3]));
    # end
    mask = np.ones(img.shape[0:3]) if mask is None else mask

    # 检测程序是否已经经过编译打包，python无效。直接设为零
    # if isdeployed
    #     parpool_flag = 0;
    # end
    parpoll_flag = 0

    # [~,~,~,ne,nrcvrs] = size(img);
    # TE1 = te(1);
    # TE2 = te(2);
    # imsize = size(img);
    ne, nrcvrs = img.shape[3:5]
    TE1 = te[0]
    TE2 = te[1]
    imsize = img.shape

    # img_diff = img(:,:,:,2,:)./img(:,:,:,1,:);
    # ph_diff = img_diff./abs(img_diff);
    # ph_diff_cmb = sum(abs(img(:,:,:,1,:)).*ph_diff,5);
    # ph_diff_cmb(isnan(ph_diff_cmb)) = 0;
    img_diff = img[:, :, :, 1, :] / img[:, :, :, 0, :]
    ph_diff = img_diff / np.abs(img_diff)
    ph_diff_cmb = np.sum(np.abs(img[:, :, :, 0, :]) * ph_diff, 4)
    ph_diff_cmb[np.isnan(ph_diff_cmb)] = 0

    # TODO 从这里开始就出问题了，源代码是直接用cd更换了目录，但是函数中不知道外接切换的目录。
    # nii = make_nii(angle(ph_diff_cmb),vox);
    # save_nii(nii,'ph_diff.nii');
    nii = nib.Nifti1Image(np.angle(ph_diff_cmb), np.eye(4))
    nib.save(nii, 'ph_diff.nii')
    # clear ph_diff img_diff
    del ph_diff, img_diff

    # % % perform unwrapping
    # % method (1)
    # % unix('prelude -p ph_diff.nii -a BET.nii -u unph_diff -m BET_mask.nii -n 12');
    # % unix('gunzip -f unph_diff.nii.gz');
    # % nii = load_nii('unph_diff.nii');
    # % unph_diff_cmb = double(nii.img);
    # mag1 = sqrt(sum(abs(img(:,:,:,1,:).^2),5));
    # mask_input = mask;
    # mask = (mag1 > 0.1*median(mag1(logical(mask(:)))));
    # mask = mask | mask_input;
    mag1 = np.sqrt(np.sum(np.abs(img[:, :, :, 0, :] ** 2), 4))
    mask_input = mask
    mask = (mag1 > 0.1 * np.median(mag1[mask.astype(bool)]))
    mask = mask | mask_input

    # % method (2)
    # % best path unwrapping
    # [pathstr, ~, ~] = fileparts(which('3DSRNCP.m'));
    # 寻找3DSRNCP的路径，这里默认文件结构固定，以Libs文件夹为分界线。
    pathstr = os.path.abspath(os.path.dirname(__file__)).split('Libs')[0]
    pathstr = os.path.join(pathstr, 'Libs', 'phase_unwrapping', '3DSRNCP')
    # setenv('pathstr',pathstr);
    # setenv('nv',num2str(imsize(1)));
    # setenv('np',num2str(imsize(2)));
    # setenv('ns',num2str(imsize(3)));
    os.environ['pathstr'] = pathstr
    os.environ['nv'] = str(imsize[0])
    os.environ['np'] = str(imsize[1])
    os.environ['ns'] = str(imsize[2])

    # fid = fopen('wrapped_phase_diff.dat','w');
    # fwrite(fid,angle(ph_diff_cmb),'float');
    # fclose(fid);
    with open('wrapped_phase_diff.dat', 'wb') as fid:
        np.angle(ph_diff_cmb).tofile(fid)

    # clear ph_diff_cmb
    del ph_diff_cmb

    # mask_unwrp = uint8(mask*255);
    mask_unwrp = (mask * 255).astype(np.uint8)
    # fid = fopen('mask_unwrp.dat','w');
    # fwrite(fid,mask_unwrp,'uchar');
    # fclose(fid);
    with open('mask_unwrp.dat', 'wb') as fid:
        fid.write(mask_unwrp.tobytes())

    # isdeployed 与python无关。
    # if isdeployed
    #     bash_script = ['~/bin/3DSRNCP wrapped_phase_diff.dat mask_unwrp.dat ' ...
    #     'unwrapped_phase_diff.dat $nv $np $ns reliability_diff.dat'];
    # else
    #     bash_script = ['${pathstr}/3DSRNCP wrapped_phase_diff.dat mask_unwrp.dat ' ...
    #     'unwrapped_phase_diff.dat $nv $np $ns reliability_diff.dat'];
    # end
    bash_script = (os.path.join(pathstr, '3DSRNCP') +
                   ' wrapped_phase_diff.dat mask_unwrp.dat'
                   ' unwrapped_phase_diff.dat $nv $np $ns reliability_diff.dat')
    # unix(bash_script);
    subprocess.run(bash_script, shell=True)

    # fid = fopen('unwrapped_phase_diff.dat','r');
    # tmp = fread(fid,'float');
    tmp = np.fromfile('unwrapped_phase_diff.dat', dtype=np.float32)
    # unph_diff_cmb = reshape(tmp - round(mean(tmp(mask==1))/(2*pi))*2*pi ,imsize(1:3)).*mask;
    unph_diff_cmb = np.reshape(
        tmp - np.round(np.mean(tmp[mask.astype(bool)]) / (2 * np.pi)) * 2 * np.pi, imsize[0:3]
    ) * mask

    # nii = make_nii(unph_diff_cmb,vox);
    # save_nii(nii,'unph_diff.nii');
    nii = nib.Nifti1Image(unph_diff_cmb, np.eye(4))
    nib.save(nii, 'unph_diff.nii')

    # % calculate initial phase offsets
    # unph_te1_cmb = unph_diff_cmb*TE1/(TE2-TE1);
    unph_diff_cmb = unph_diff_cmb * TE1 / (TE2 - TE1)
    # offsets = img(:,:,:,1,:)./exp(1j*unph_te1_cmb);
    offsets = img[:, :, :, 0, :] / np.exp(1j * unph_diff_cmb)
    # offsets = offsets./abs(offsets); % complex phase offset
    offsets = offsets / np.abs(offsets)  # complex phase offset
    # offsets(isnan(offsets)) = 0;
    offsets[np.isnan(offsets)] = 0

    # nii = make_nii(angle(offsets),vox);
    # save_nii(nii,'offsets_raw.nii');
    nii = nib.Nifti1Image(np.angle(offsets), np.eye(4))
    nib.save(nii, 'offsets_raw.nii')

    # % smooth offsets
    # % maybe later change to smooth the real and imag parts seperately, and try
    # % guassian filter!
    # if strcmpi('smooth3',smooth_method)
    #   for chan = 1:nrcvrs
    #     offsets(:,:,:,1,chan) = smooth3(offsets(:,:,:,1,chan),'box',round(5)*2+1);
    #     offsets(:,:,:,1,chan) = smooth3(offsets(:,:,:,1,chan),'box',round(2./vox)*2+1);
    #     offsets(:,:,:,1,chan) = offsets(:,:,:,1,chan)./abs(offsets(:,:,:,1,chan));
    #   end
    if smooth_method.lower() == "smooth3":
        for chan in range(nrcvrs):
            offsets[:, :, :, 0, chan] = ndimage.uniform_filter(offsets[:, :, :, 0, chan], size=round(5) * 2 + 1)
            offsets[:, :, :, 0, chan] = ndimage.uniform_filter(offsets[:, :, :, 0, chan], size=round(2 / vox) * 2 + 1)
            offsets[:, :, :, 0, chan] = offsets[:, :, :, 0, chan] / np.abs(offsets[:, :, :, 0, chan])

    # elseif strcmpi('gaussian',smooth_method)
    #   for chan = 1:nrcvrs
    #       offsets(:,:,:,1,chan) = imgaussfilt3(real(offsets(:,:,:,1,chan)),6) + 1j*imgaussfilt3(imag(offsets(:,:,:,1,chan)),6);
    #       offsets(:,:,:,1,chan) = offsets(:,:,:,1,chan)./abs(offsets(:,:,:,1,chan));
    #   end
    elif smooth_method.lower() == "gaussian":
        for chan in range(nrcvrs):
            offsets[:, :, :, 0, chan] = (ndimage.gaussian_filter(np.real(offsets[:, :, :, 0, chan]), sigma=6) +
                                         1j * ndimage.gaussian_filter(np.imag(offsets[:, :, :, 0, chan]), sigma=6))
            offsets[:, :, :, 0, chan] = offsets[:, :, :, 0, chan] / np.abs(offsets[:, :, :, 0, chan])

    # elseif strcmpi('poly3',smooth_method)
    #     for chan = 1:nrcvrs
    #         fid = fopen(['wrapped_offsets_chan' num2str(chan) '.dat'],'w');
    #         fwrite(fid,angle(offsets(:,:,:,chan)),'float');
    #         fclose(fid);
    #         setenv('chan',num2str(chan));

    #         bash_script = ['${pathstr}/3DSRNCP wrapped_offsets_chan${chan}.dat mask_unwrp.dat
    #           'unwrapped_offsets_chan${chan}.dat $nv $np $ns reliability_diff.dat'];

    #         unix(bash_script) ;
    #         fid = fopen(['unwrapped_offsets_chan' num2str(chan) '.dat'],'r');
    #         tmp = fread(fid,'float');
    #         unph_offsets(:,:,:,chan) = reshape(tmp - round(mean(tmp(mask==1))/(2*pi))*2*pi ,imsize(1:3)).*mask;
    #         fclose(fid);
    #         offsets(:,:,:,chan) = poly3d(unph_offsets(:,:,:,chan),mask,3);
    #     end
    #     offsets = exp(1j*offsets);
    elif smooth_method.lower() == 'poly3':
        # TODO unph_offsets 大概和offsets一样大？
        unph_offsets = np.zeros_like(offsets)
        for chan in range(nrcvrs):
            # 保存为二进制文件
            # np.save('wrapped_offsets_chan{}.dat'.format(chan), np.angle(offsets[..., 0, chan])) np.save 保存的是npy文件
            with open(f'wrapped_offsets_chan{str(chan)}.dat', 'wb') as fid:
                fid.write(np.angle(offsets).astype('float32').tobytes())
            os.environ['chan'] = str(chan)
            # 运行3DSRNCP
            bash_script = (pathstr + f'/3DSRNCP wrapped_offsets_chan{str(chan)}.dat mask_unwrp.dat '
                                          f'unwrapped_offsets_chan{str(chan)}.dat $nv $np $ns reliability_diff.dat')
            # unix(bash_script)
            subprocess.run(bash_script, shell=True)

            # TODO 读取unwrapped_offsets_chan${chan}.dat
            with open(f'unwrapped_offsets_chan{str(chan)}.dat', 'rb') as fid:
                tmp = np.fromfile(fid, dtype='float32')
            tmp = np.reshape(tmp - np.round(np.mean(tmp[mask == 1]) / (2 * np.pi)) * 2 * np.pi, imsize[0:3])
            unph_offsets[..., chan] = tmp * mask
            offsets[..., chan] = poly3d(unph_offsets[..., chan], mask, 3)

        # offsets = exp(1j*offsets)
        offsets = np.exp(1j * offsets)

    # elseif strcmpi('poly3_nlcg',smooth_method)
    #     for chan = 1:nrcvrs
    #         offsets(:,:,:,:,chan) = poly3d_nonlinear(offsets(:,:,:,:,chan),mask,3);
    #     end
    elif smooth_method.lower() == 'poly3_nlcg':
        for chan in range(nrcvrs):
            offsets[..., chan] = poly3d_nonlinear(offsets[..., chan], mask, 3)
    # else
    #     error('what method to use for smoothing? smooth3 or poly3 or poly3_nlcg')
    # end
    else:
        raise ValueError('what method to use for smoothing? smooth3 or poly3 or poly3_nlcg')

    # nii = make_nii(angle(offsets),vox);
    # save_nii(nii,'offsets_smooth.nii');
    nii = nib.Nifti1Image(np.angle(offsets), np.eye(4))
    nib.save(nii, 'offsets_smooth.nii.gz')

    # % combine phase according to complex summation
    # % offsets = repmat(offsets,[1,1,1,ne,1]);
    # img = img./offsets;
    # img(isnan(img)) = 0;
    # ph_cmb = angle(sum(img,5));
    # ph_cmb(isnan(ph_cmb)) = 0;
    # mag_cmb = abs(mean(img,5));
    # mag_cmb(isnan(mag_cmb)) = 0;
    img = img / offsets
    img[np.isnan(img)] = 0
    ph_cmb = np.angle(np.sum(img, axis=4))
    ph_cmb[np.isnan(ph_cmb)] = 0
    mag_cmb = np.abs(np.mean(img, axis=4))
    mag_cmb[np.isnan(mag_cmb)] = 0

    # % sen = squeeze(abs(img(:,:,:,1,:)))./repmat(mag_cmb(:,:,:,1),[1 1 1 nrcvrs]);
    # sen = squeeze(abs(img(:,:,:,1,:)))./mag_cmb(:,:,:,1);
    # sen(isnan(sen)) = 0;
    # sen(isinf(sen)) = 0;
    # nii = make_nii(sen,vox);
    # save_nii(nii,'sen_mag_raw.nii');
    sen = np.squeeze(np.abs(img[..., 0, :])) / mag_cmb[..., 0]
    sen[np.isnan(sen)] = 0
    sen[np.isinf(sen)] = 0
    nii = nib.Nifti1Image(sen, np.eye(4))
    nib.save(nii, 'sen_mag_raw.nii.gz')


    # % smooth the coil sensitivity
    # for chan = 1:nrcvrs
    #     % sen_smooth(:,:,:,chan) = smooth3(sen(:,:,:,chan),'box',round(8)*2+1);
    #     sen(:,:,:,chan) = imgaussfilt3(real(sen(:,:,:,chan)),4);
    # end
    for chan in range(nrcvrs):
        # sen(:,:,:,chan) = imgaussfilt3(real(sen(:,:,:,chan)),4);
        sen[..., chan] = ndimage.gaussian_filter(np.real(sen[..., chan]), 4)


    # nii = make_nii(sen,vox);
    # save_nii(nii,'sen_mag_smooth.nii');
    nii = nib.Nifti1Image(sen, np.eye(4))
    nib.save(nii, 'sen_mag_smooth.nii.gz')

    # coil_sens = sen.*squeeze(offsets);
    coil_sens = sen * np.squeeze(offsets)
