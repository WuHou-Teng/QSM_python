import os
import struct
import numpy as np
from scipy import ndimage
from Libs.Misc.NIFTI_python.nifti_base import make_save_nii_engine
from Libs.background_field_removal.poly3d import poly3d
from Libs.background_field_removal.poly3d_nonlinear import poly3d_nonlinear


def poem(mag, pha, vox, te, mask=None, smooth_method=None, parpool_flag=None):
    """
    Gradient-echo multi-echo combination (for phase).
      PH_CMB = POEM(MAG, PHA, VOX, TE, MASK, SMOOTH_METHOD) combines phase from multiple receivers
        MAG/PHA:     raw complex images from multiple receivers, 5D: [3D_image, echoes, receiver channels]
      TE :     echo times
      MASK:    brain mask
      VOX:     spatial resolution/voxel size, e.g. [1 1 1] for isotropic
      PH_CMB:  phase after combination
      MAG_CMB: magnitude after combination
      SMOOTH:  smooth methods(1) smooth3, (2) poly3, (3) poly3_nlcg, (4) gaussian
    :param mag: 来自多个接收器的原始复杂图像，5D: [3D_image，回波，接收器通道]
    :param pha: 来自多个接收器的原始复杂图像，5D: [3D_image，回波，接收器通道]
    :param vox: spatial resolution/voxel size, e.g. [1 1 1] for isotropic
    :param te: echo times
    :param mask: brain mask
    :param smooth_method: smooth methods(1) smooth3, (2) poly3, (3) poly3_nlcg, (4) gaussian
    :param parpool_flag:
    :return: ph_cmb, mag_cmb, coil_sens
    """
    # if ~ exist('mask','var') || isempty(mask)
    # 		mask = ones(size(mag,[1,2,3]))
    # 	end
    if mask is None:
        mask = np.ones(mag.shape[0:3])

    if smooth_method is None:
        smooth_method = 'gaussian'

    # 并行运算的标志，python中因为GIL的存在，使用多线程并不能真正地实现并行化处理
    if parpool_flag is None:
        parpool_flag = 0

    # isdeployed 用于检测matlab是否在独立应用中运行, 在python中无效。
    isdeployed = True
    # if isdeployed:
    #     parpool_flag = 0

    # 	[~,~,~,~,nrcvrs] = size(mag)
    nrcvrs = mag.shape[4]

    TE1 = te[0]
    TE2 = te[1]

    # 	imsize = size(mag)
    imsize = np.shape(mag)

    # 	ph_diff = exp(1j*pha(:,:,:,2,:))./exp(1j*pha(:,:,:,1,:))
    ph_diff = np.exp(1j * pha[:, :, :, 1, :]) / np.exp(1j * pha[:, :, :, 0, :])

    # 	ph_diff_cmb = sum(mag.*ph_diff,5)
    ph_diff_cmb = np.sum(mag * ph_diff, axis=4)

    # 	ph_diff_cmb(isnan(ph_diff_cmb)) = 0
    ph_diff_cmb[np.isnan(ph_diff_cmb)] = 0

    # 	nii = make_nii(angle(ph_diff_cmb),vox)
    # 	save_nii(nii,'ph_diff.nii')
    make_save_nii_engine(np.angle(ph_diff_cmb), "ph_diff_cmb", vox, 'ph_diff.nii')

    # 	clear ph_diff
    del ph_diff

    # % perform unwrapping
    # method (1)
    # unix('prelude -p ph_diff.nii -a BET.nii -u unph_diff -m BET_mask.nii -n 12');
    # unix('gunzip -f unph_diff.nii.gz');
    # nii = load_nii('unph_diff.nii');
    # unph_diff_cmb = double(nii.img);

    # 	mag1 = sqrt(sum((mag(:,:,:,1,:).^2),5))
    mag1 = np.sqrt(np.sum((mag[:, :, :, 0, :] ** 2), axis=4))
    mask_input = mask
    # 	mask = (mag1 > 0.1*median(mag1(logical(mask(:)))))
    mask = (mag1 > 0.1 * np.median(mag1[mask.astype(bool)]))
    # TODO 这里的astype并不一定要加
    assert type(mask) == np.ndarray
    mask = mask.astype(int)
    # 	mask = mask | mask_input
    mask = mask | mask_input

    # method (2)
    # best path unwrapping

    # [pathstr, ~, ~] = fileparts(which('3DSRNCP.m'))
    # file = 'H:\user4\matlab\myfile.txt';
    # [filepath,name,ext] = fileparts(file)
    # << filepath = 'H:\user4\matlab'
    # << name = 'myfile'
    # << ext = '.txt'
    # 寻找'3DSRNCP.m'所在的路径, 这里直接将路径写死
    path_3dSRNCP = os.getcwd() + "\\..\\phase_unwrap\\"

    # setenv('pathstr',pathstr)
    # setenv('nv',num2str(imsize(1)))
    # setenv('np',num2str(imsize(2)))
    # setenv('ns',num2str(imsize(3)))
    # 设定环境变量
    os.environ['path_3dSRNCP'] = str(path_3dSRNCP)
    os.environ['nv'] = str(imsize[0])
    os.environ['np'] = str(imsize[1])
    os.environ['ns'] = str(imsize[2])

    # fid = fopen('wrapped_phase_diff.dat','w')
    # fwrite(fid,angle(ph_diff_cmb),'float')
    # fclose(fid)
    # TODO 将ph_diff_cmb写入文件, 以便于3DSRNCP读取，但不知道这种写法是否正确
    with open('wrapped_phase_diff.dat', 'wb') as fid:
        np.array(np.angle(ph_diff_cmb), dtype=np.float32).tofile(fid)
        # for val in np.angle(ph_diff_cmb).flatten():
        #     fid.write(struct.pack('f', val))

    del ph_diff_cmb

    # 	mask_unwrp = uint8(mask*255)
    mask_unwrp = mask.astype(np.uint8) * 255

    # 	fid = fopen('mask_unwrp.dat','w')
    # 	fwrite(fid,mask_unwrp,'uchar')
    # 	fclose(fid)
    # TODO 此处的书写格式是 uchar，我不确定改为python后，是不是就什么也不用改。
    with open('wrapped_phase_diff.dat', 'wb') as fid:
        fid.write(mask_unwrp.tobytes())
        # for val in mask_unwrp.flatten():
        #     fid.write(val)
    #
    # if isdeployed
    #   bash_script = ['~/bin/3DSRNCP wrapped_phase_diff.dat mask_unwrp.dat \
    #   unwrapped_phase_diff.dat $nv $np $ns reliability_diff.dat'];
    # else
    #   bash_script = ['${pathstr}/3DSRNCP wrapped_phase_diff.dat mask_unwrp.dat \
    #   unwrapped_phase_diff.dat $nv $np $ns reliability_diff.dat'];
    # end
    # 	unix(bash_script)
    bash_script = (path_3dSRNCP + '3DSRNCP wrapped_phase_diff.dat mask_unwrp.dat '
                             'unwrapped_phase_diff.dat $nv $np $ns reliability_diff.dat')
    os.system(bash_script)

    # fid = fopen('unwrapped_phase_diff.dat','r')
    # tmp = fread(fid,'float')
    # TODO 不确定这里的读取是否正确
    with open('unwrapped_phase_diff.dat', 'rb') as fid:
        tmp = np.fromfile(fid, dtype=np.float32)

    # unph_diff_cmb = reshape(tmp - round(mean(tmp(mask==1))/(2*pi))*2*pi ,imsize(1:3)).*mask
    unph_diff_cmb = np.reshape(
        tmp - np.round(np.mean(tmp[mask.astype(bool)]) / (2 * np.pi)) * 2 * np.pi, imsize[0:3]
    ) * mask

    # 	nii = make_nii(unph_diff_cmb,vox)
    # 	save_nii(nii,'unph_diff.nii')
    make_save_nii_engine(unph_diff_cmb, "unph_diff_cmb", vox, 'unph_diff.nii')

    # calculate initial phase offsets
    # unph_te1_cmb = unph_diff_cmb*TE1/(TE2-TE1)
    unph_te1_cmb = unph_diff_cmb * TE1 / (TE2 - TE1)

    # # offsets = exp(1j*pha(:,:,:,1,:))./repmat(exp(1j*unph_te1_cmb),[1,1,1,1,nrcvrs]);
    # offsets = exp(1j*pha(:,:,:,1,:))./exp(1j*unph_te1_cmb)
    offsets = np.exp(1j * pha[:, :, :, 0, :]) / np.exp(1j * unph_te1_cmb)

    # offsets(isnan(offsets)) = 0
    offsets[np.isnan(offsets)] = 0

    # nii = make_nii(angle(offsets),vox)
    # save_nii(nii,'offsets_raw.nii')
    make_save_nii_engine(np.angle(offsets), "offsets_raw", vox, 'offsets_raw.nii')

    # smooth offsets
    # maybe later change to smooth the real and imag parts seperately, and try
    # guassian filter!
    # 	if strcmpi('smooth3',smooth_method)
    # 		if parpool_flag
    # 			parpool
    # 			parfor chan = 0:nrcvrs
    # 			offsets(:,:,:,1,chan) = smooth3(offsets(:,:,:,1,chan),'box',round(5)*2+1)
    # 			#       offsets(:,:,:,1,chan) = smooth3(offsets(:,:,:,1,chan),'box',round(2./vox)*2+1);
    # 			offsets(:,:,:,1,chan) = offsets(:,:,:,1,chan)./abs(offsets(:,:,:,1,chan))
    # 		end
    # 		delete(gcp('nocreate'))
    # 	else
    # 		for chan = 1:nrcvrs
    # 			offsets(:,:,:,1,chan) = smooth3(offsets(:,:,:,1,chan),'box',round(5)*2+1)
    # 			#       offsets(:,:,:,1,chan) = smooth3(offsets(:,:,:,1,chan),'box',round(2./vox)*2+1);
    # 			offsets(:,:,:,1,chan) = offsets(:,:,:,1,chan)./abs(offsets(:,:,:,1,chan))
    # 		end
    # 	end

    if smooth_method.lower() == 'smooth3':
        # python中不使用并行运算，所以这里的parpool_flag不需要

        for chan in range(nrcvrs):
            # 由于原代码采用的是"box" filter，所以这里使用uniform_filter
            offsets[..., 0, chan] = ndimage.uniform_filter(offsets[..., 0, chan], size=round(5) * 2 + 1)
            offsets[..., 0, chan] /= np.abs(offsets[..., 0, chan])

    # elseif strcmpi('gaussian',smooth_method)
    # 	if parpool_flag
    # 		parpool
    # 		parfor chan = 0:nrcvrs
    # 		offsets(:,:,:,1,chan) = imgaussfilt3(real(offsets(:,:,:,1,chan)),6) +
    # 	                           	1j*imgaussfilt3(imag(offsets(:,:,:,1,chan)),6)
    # 		offsets(:,:,:,1,chan) = offsets(:,:,:,1,chan)./abs(offsets(:,:,:,1,chan))
    # 	end
    # 	delete(gcp('nocreate'))
    #   else
    # 	    for chan = 0:nrcvrs
    # 	    	offsets(:,:,:,1,chan) = imgaussfilt3(real(offsets(:,:,:,1,chan)),6) +
    # 	                            	1j*imgaussfilt3(imag(offsets(:,:,:,1,chan)),6)
    # 	    	offsets(:,:,:,1,chan) = offsets(:,:,:,1,chan)./abs(offsets(:,:,:,1,chan))
    # 	    end
    # end
    elif smooth_method.lower() == 'gaussian':
        for chan in range(nrcvrs):
            offsets[..., 0, chan] = ndimage.gaussian_filter(offsets[..., 0, chan].real, sigma=6) + \
                                    1j * ndimage.gaussian_filter(offsets[..., 0, chan].imag, sigma=6)
            offsets[..., 0, chan] /= np.abs(offsets[..., 0, chan])

    # elseif strcmpi('poly3',smooth_method)
    # 	for chan = 1:nrcvrs
    # 		fid = fopen(['wrapped_offsets_chan' num2str(chan) '.dat'],'w')
    # 		fwrite(fid,angle(offsets(:,:,:,chan)),'float')
    # 		fclose(fid)

    # 		setenv('chan',num2str(chan))

    # 		if isdeployed
    # 			bash_script = ['~/bin/3DSRNCP wrapped_offsets_chan${chan}.dat mask_unwrp.dat
    # 			unwrapped_offsets_chan${chan}.dat $nv $np $ns reliability_diff.dat'];
    # 		else
    # 			bash_script = ['${pathstr}/3DSRNCP wrapped_offsets_chan${chan}.dat mask_unwrp.dat ' ...
    #             'unwrapped_offsets_chan${chan}.dat $nv $np $ns reliability_diff.dat'];
    # 		end
    # 		unix(bash_script)
    # 		fid = fopen(['unwrapped_offsets_chan' num2str(chan) '.dat'],'r')
    # 		tmp = fread(fid,'float')
    # 		unph_offsets(:,:,:,chan) = reshape(tmp - round(mean(tmp(mask==1))/(2*pi))*2*pi ,imsize(0:3)).*mask
    # 		fclose(fid)
    # 		offsets(:,:,:,chan) = poly3d(unph_offsets(:,:,:,chan),mask,3)
    # 	end
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
            bash_script = (path_3dSRNCP + f'/3DSRNCP wrapped_offsets_chan{str(chan)}.dat mask_unwrp.dat '
                                          f'unwrapped_offsets_chan{str(chan)}.dat $nv $np $ns reliability_diff.dat')
            # unix(bash_script)
            os.system(bash_script)

            # TODO 读取unwrapped_offsets_chan${chan}.dat
            with open(f'unwrapped_offsets_chan{str(chan)}.dat', 'rb') as fid:
                tmp = np.fromfile(fid, dtype='float32')
            tmp = np.reshape(tmp - np.round(np.mean(tmp[mask == 1]) / (2 * np.pi)) * 2 * np.pi, imsize[0:3])
            unph_offsets[..., chan] = tmp * mask
            # TODO 改写 poly3d 为python。
            offsets[..., chan] = poly3d(unph_offsets[..., chan], mask, 3)

        # offsets = exp(1j*offsets)
        offsets = np.exp(1j * offsets)

    # elseif strcmpi('poly3_nlcg',smooth_method)
    # 	for chan = 0:nrcvrs
    # 		offsets(:,:,:,:,chan) = poly3d_nonlinear(offsets(:,:,:,:,chan),mask,3)
    # 	end
    elif smooth_method.lower() == 'poly3_nlcg':
        for chan in range(nrcvrs):
            # TODO poly3d_nonlinear 改写
            offsets[..., chan] = poly3d_nonlinear(offsets[..., chan], mask, 3)
    # else
    # 	error('what method to use for smoothing? smooth3 or poly3 or poly3_nlcg')
    # end
    else:
        raise ValueError('what method to use for smoothing? smooth3 or poly3 or poly3_nlcg')

    # nii = make_nii(angle(offsets),vox)
    # save_nii(nii,'offsets_smooth.nii')
    make_save_nii_engine(np.angle(offsets), "offsets",vox, 'offsets_smooth.nii')

    # # combine phase according to complex summation
    # img_cmb = mean(mag.*exp(1j*pha)./offsets,5)
    # img_cmb(isnan(img_cmb)) = 0
    # ph_cmb = angle(img_cmb)
    # ph_cmb(isnan(ph_cmb)) = 0
    # mag_cmb = abs(img_cmb)
    # mag_cmb(isnan(mag_cmb)) = 0
    img_cmb = np.mean(mag * np.exp(1j * pha) / offsets, axis=4)
    img_cmb[np.isnan(img_cmb)] = 0
    ph_cmb = np.angle(img_cmb)
    ph_cmb[np.isnan(ph_cmb)] = 0
    mag_cmb = np.abs(img_cmb)
    mag_cmb[np.isnan(mag_cmb)] = 0

    # clear img_cmb
    del img_cmb

    # # sen = squeeze((mag(:,:,:,1,:)))./repmat(mag_cmb(:,:,:,1),[1 1 1 nrcvrs]);
    # sen = squeeze((mag(:,:,:,1,:)))./mag_cmb(:,:,:,1)
    sen = np.squeeze((mag[..., 0, :])) / mag_cmb[..., 0]
    # sen(isnan(sen)) = 0
    # sen(isinf(sen)) = 0
    sen[np.isnan(sen)] = 0
    sen[np.isinf(sen)] = 0
    # nii = make_nii(sen,vox)
    # save_nii(nii,'sen_mag_raw.nii')
    make_save_nii_engine(sen, "sen", 'sen_mag_raw.nii', vox)

    # # smooth the coil sensitivity
    # for chan = 0:nrcvrs
    # 	# sen_smooth(:,:,:,chan) = smooth3(sen(:,:,:,chan),'box',round(8)*2+1);
    # 	sen(:,:,:,chan) = imgaussfilt3(real(sen(:,:,:,chan)),4)
    # end
    for chan in range(nrcvrs):
        # sen_smooth[..., chan] = smooth3(sen[..., chan], 'box', round(8) * 2 + 1)
        sen[..., chan] = ndimage.gaussian_filter(np.real(sen[..., chan]), sigma=round(8) * 2 + 1)

    # nii = make_nii(sen,vox)
    # save_nii(nii,'sen_mag_smooth.nii')
    make_save_nii_engine(sen, "sen", 'sen_mag_smooth.nii', vox)

    # coil_sens = sen.*squeeze(offsets)
    coil_sens = sen * np.squeeze(offsets)

    return [ph_cmb, mag_cmb, coil_sens]
