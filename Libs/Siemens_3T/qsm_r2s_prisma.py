import os
import subprocess

from scipy import ndimage
from scipy.io import savemat, loadmat
import pydicom
import numpy as np
import nibabel as nib

from Libs.Misc.shaver import shaver
from Libs.Varian_4p7T.echofit import echofit
from Libs.background_field_removal.poly3d import poly3d
from Libs.background_field_removal.projection_onto_dipole_fields import projectionontodipolefields
from Libs.background_field_removal.resharp import resharp
from Libs.background_field_removal.sharp import sharp
from Libs.coil_combination.geme_cmb import geme_cmb
from Libs.dipole_inversion.tikhonov_qsm import tikhonov_qsm
from Libs.dipole_inversion.tvdi import tvdi
from tools.Exception_Def import ParamException
from tools.StructureCLass import Options, Param
from Transfer_Tools.numpy_over_write import padarray
from Libs.Misc.NIFTI_python.nifti_base import make_save_nii_engine


def V_SHARP(tfs, param, param1, smvsize, param2, vox):
    return 0, 0

def extendharmonicfield(reducedBackgroundField, mask, maskReduced, Parameters):
    return 0


def QSM_iLSQR(param, mask_resharp, param1, z_prjs, param2, vox, param3, param4, param5, param6, param7,
              MagneticFieldStrength):
    pass


def LBV(tfs, param, param1, vox, lbv_tol, lbv_peel):
    pass


def qsm_r2s_prisma(path_mag=None, path_ph=None, path_out=None, options=None):
    """
    QSM_R2S_PRISMA Quantitative susceptibility mapping from R2s sequence at PRISMA (3T).
    QSM_R2S_PRISMA(PATH_MAG, PATH_PH, PATH_OUT, OPTIONS) reconstructs susceptibility maps.
    Re-define the following default settings if necessary
    :param path_mag: - directory of magnitude dicoms
    :param path_ph:  - directory of unfiltered phase dicoms
    :param path_out: - directory to save nifti and/or matrixes   : QSM_SWI_PRISMA
    :param options: Options 类，可供调节的参数如下：

    OPTIONS      - parameter structure including fields below
        .readout    - multi-echo 'unipolar' or 'bipolar'        : 'unipolar'
        .r_mask     - whether to enable the extra masking       : 1
        .fit_thr    - extra filtering based on the fit residual : 40
        .bet_thr    - threshold for BET brain mask              : 0.4
        .bet_smooth - smoothness of BET brain mask at edges     : 2
        .ph_unwrap  - 'prelude' or 'bestpath'                   : 'prelude'
        .bkg_rm     - background field removal method(s)        : 'resharp'
                      options: 'pdf','sharp','resharp','esharp','lbv'
                      to try all e.g.: {'pdf','sharp','resharp','esharp','lbv'}
                      set to 'lnqsm' or 'tfi' for LN-QSM and TFI methods
        .t_svd      - truncation of SVD for SHARP               : 0.1
        .smv_rad    - radius (mm) of SMV convolution kernel     : 3
        .tik_reg    - Tikhonov regularization for resharp       : 1e-3
        .cgs_num    - max interation number for RESHARP         : 500
        .lbv_peel   - LBV layers to be peeled off               : 2
        .lbv_tol    - LBV interation error tolerance            : 0.01
        .tv_reg     - Total variation regularization parameter  : 5e-4
        .tvdi_n     - iteration number of TVDI (nlcg)           : 500
    :return:
    """
    if path_mag is None or not os.path.exists(path_mag):
        raise ParamException("", "Please input the directory of magnitude DICOMs")

    # if ~ exist('path_mag','var') || isempty(path_mag)
    #     error('Please input the directory of magnitude DICOMs')
    # end
    if path_ph is None or not os.path.exists(path_ph):
        raise ParamException("", "Please input the directory of unfiltered phase DICOMs")

    # if ~ exist('path_ph','var') || isempty(path_ph)
    #     error('Please input the directory of unfiltered phase DICOMs')
    # end
    if path_ph is None or not os.path.exists(path_ph):
        raise ParamException("", "Please input the directory of unfiltered phase DICOMs")

    # if ~ exist('path_out','var') || isempty(path_out)
    #     path_out = pwd;
    #     disp('Current directory for output')
    # end
    if path_out is None or not os.path.exists(path_out):
        path_out = os.getcwd()
        print('Current directory for output')

    # if ~ exist('options','var') || isempty(options)
    #     options = [];
    # end
    if options is None:
        options = Options()

    # if ~ isfield(options,'readout')
    #     options.readout = 'unipolar';
    # end
    if not options.is_field('readout'):
        options.readout = 'unipolar'

    # if ~ isfield(options,'r_mask')
    #     options.r_mask = 1;
    # end
    if not options.is_field('r_mask'):
        options.r_mask = 1

    # if ~ isfield(options,'fit_thr')
    #     options.fit_thr = 40;
    # end
    if not options.is_field('fit_thr'):
        options.fit_thr = 40

    # if ~ isfield(options,'bet_thr')
    #     options.bet_thr = 0.4;
    # end
    if not options.is_field('bet_thr'):
        options.bet_thr = 0.4

    # if ~ isfield(options,'bet_smooth')
    #     options.bet_smooth = 2;
    # end
    if not options.is_field('bet_smooth'):
        options.bet_smooth = 2

    # if ~ isfield(options,'ph_unwrap')
    #     options.ph_unwrap = 'bestpath';
    # end
    if not options.is_field('ph_unwrap'):
        options.ph_unwrap = 'bestpath'

    # if ~ isfield(options,'bkg_rm')
    #     options.bkg_rm = 'resharp';
    #     % options.bkg_rm = {'pdf','sharp','resharp','esharp','lbv'};
    # end
    if not options.is_field('bkg_rm'):
        options.bkg_rm = 'resharp'
        # options.bkg_rm = {'pdf','sharp','resharp','esharp','lbv'};

    if not options.is_field('t_svd'):
        options.t_svd = 0.1

    if not options.is_field('smv_rad'):
        options.smv_rad = 3

    if not options.is_field('tik_reg'):
        options.tik_reg = 1e-3

    if not options.is_field('cgs_num'):
        options.cgs_num = 500

    if not options.is_field('lbv_tol'):
        options.lbv_tol = 0.01

    if not options.is_field('lbv_peel'):
        options.lbv_peel = 2

    if not options.is_field('tv_reg'):
        options.tv_reg = 5e-4

    if not options.is_field('inv_num'):
        options.inv_num = 500

    if not options.is_field('interp'):
        options.interp = 0

    readout = options.readout
    r_mask = options.r_mask
    fit_thr = options.fit_thr
    bet_thr = options.bet_thr
    bet_smooth = options.bet_smooth
    ph_unwrap = options.ph_unwrap
    bkg_rm = options.bkg_rm
    t_svd = options.t_svd
    smv_rad = options.smv_rad
    tik_reg = options.tik_reg
    cgs_num = options.cgs_num
    lbv_tol = options.lbv_tol
    lbv_peel = options.lbv_peel
    tv_reg = options.tv_reg
    inv_num = options.inv_num
    interp = options.interp

    # % read in DICOMs of both magnitude and raw unfiltered phase images
    # % read in magnitude DICOMs
    # path_mag = cd(cd(path_mag)); TODO 这里直接跳到对应的文件夹，避免了后面的路径拼接
    os.chdir(path_mag)
    # mag_list = dir(path_mag);
    mag_list = os.listdir(path_mag)  # 获取目录下所有文件的名称列表。
    # mag_list = mag_list(~strncmpi('.', {mag_list.name}, 1));
    mag_list = [filename for filename in mag_list if not filename.startswith('.')]  # 去除隐藏文件

    # % get the sequence parameters
    # dicom_info = dicominfo([path_mag,filesep,mag_list(end).name]);
    dicom_info = pydicom.dcmread(os.path.join(path_mag, mag_list[-1]))  # 读取dicom文件
    EchoTrainLength = int(dicom_info.EchoNumbers)  # 获取回波数
    # for i = 1:EchoTrainLength % read in TEs
    #     dicom_info = dicominfo([path_mag,filesep,mag_list(1+(i-1)*(length(mag_list))./EchoTrainLength).name]);
    #     TE(dicom_info.EchoNumbers) = dicom_info.EchoTime*1e-3;
    # end
    # 读取所有的TE,
    TE = [0] * EchoTrainLength
    for i in range(EchoTrainLength):
        try:
            dicom_info = pydicom.dcmread(
                os.path.join(path_mag, mag_list[np.ceil(1 + i * len(mag_list) / EchoTrainLength)]))
        except IndexError:
            dicom_info = pydicom.dcmread(
                os.path.join(path_mag, mag_list[np.floor(1 + i * len(mag_list) / EchoTrainLength)]))
        # 这里涉及到matlab中数组长度的动态更新，python数组需要预定义，具体长度未知。所以采用手动动态更新。
        try:
            TE[dicom_info.EchoNumbers] = dicom_info.EchoTime * 1e-3
        except IndexError:
            for j in range(dicom_info.EchoNumbers - len(TE) + 1):
                TE.append(0)
            TE[dicom_info.EchoNumbers] = dicom_info.EchoTime * 1e-3
    TE = np.array(TE)
    # vox = [dicom_info.PixelSpacing(1), dicom_info.PixelSpacing(2), dicom_info.SliceThickness];
    vox = [dicom_info.PixelSpacing[0], dicom_info.PixelSpacing[1], dicom_info.SliceThickness]

    # % angles!!! (z projections)
    # Xz = dicom_info.ImageOrientationPatient(3);
    Xz = dicom_info.ImageOrientationPatient[2]
    # Yz = dicom_info.ImageOrientationPatient(6);
    Yz = dicom_info.ImageOrientationPatient[5]
    # % Zz = sqrt(1 - Xz^2 - Yz^2);
    # Zxyz = cross(dicom_info.ImageOrientationPatient(1:3),dicom_info.ImageOrientationPatient(4:6));
    Zxyz = np.cross(dicom_info.ImageOrientationPatient[0:3], dicom_info.ImageOrientationPatient[3:6])
    # Zz = Zxyz(3);
    Zz = Zxyz[2]
    # z_prjs = [Xz, Yz, Zz];
    z_prjs = np.array([Xz, Yz, Zz])

    # for i = 1:length(mag_list)
    #     [NS,NE] = ind2sub([length(mag_list)./EchoTrainLength,EchoTrainLength],i);
    #     mag(:,:,NS,NE) = permute(single(dicomread([path_mag,filesep,mag_list(i).name])),[2,1]);
    # end
    # mag的维度为[Columns, Rows, NS, NE]，其中Columns和Rows和matlab是反过来的。此外，NS为slice方向，NE为echo方向。
    mag = np.zeros([dicom_info.Columns, dicom_info.Rows, len(mag_list) / EchoTrainLength, EchoTrainLength])
    for i in range(len(mag_list)):
        # matlab 的ind2sub 和 numpy的unravel_index的效果有所区别
        # matlab:
        #   >> array = [3, 4, 5, 6]
        #   >> [row, col] = ind2sub([3 3], array)
        #   << row = [3, 1, 2, 3], col = [1, 2, 2, 2]
        # numpy:
        #   >> array = [2, 3, 4, 5] (索引方面同样因为初始值，python需要减一）
        #   >> row, col = np.unravel_index(array, [3, 3])
        #   << row = [0, 1, 1, 1], col = [2， 0， 1， 2] ## 注意，numpy 反馈的 col 和 row 和 matlab 是反过来的。
        # 获取索引i对应的，在[len(mag_list) / EchoTrainLength, EchoTrainLength]中的位置坐标。
        NE, NS = np.unravel_index(i, [len(mag_list) / EchoTrainLength, EchoTrainLength])  # 将NS和NE对调。
        # 在NumPy中，.pixel_array是DICOM（数字图像通信）图像对象的属性之一。
        # DICOM是医学图像和相关信息的国际标准，.pixel_array用于存储DICOM图像的像素数据。
        mag[:, :, NS, NE] = np.transpose(np.single(pydicom.dcmread(os.path.join(path_mag, mag_list[i])).pixel_array))
    # % size of matrix
    # imsize = size(mag);
    imsize = mag.shape

    # % read in phase DICOMs
    # path_ph = cd(cd(path_ph));
    os.chdir(path_ph)  # 还是采用cd，不然调用的函数不知道自己的工作文件夹
    # ph_list = dir(path_ph);
    ph_list = os.listdir(path_ph)
    # ph_list = ph_list(~strncmpi('.', {ph_list.name}, 1));
    ph_list = [i for i in ph_list if not i.startswith('.')]  # 去除隐藏文件
    #
    # for i = 1:length(ph_list)
    #     [NS,NE] = ind2sub([length(ph_list)./EchoTrainLength,EchoTrainLength],i);
    #     ph(:,:,NS,NE) = permute(single(dicomread([path_ph,filesep,ph_list(i).name])),[2,1]);% covert to [-pi pi] range
    #     ph(:,:,NS,NE) = ph(:,:,NS,NE)/4095*2*pi - pi;
    # end
    ph = np.zeros([dicom_info.Columns, dicom_info.Rows, len(ph_list) / EchoTrainLength, EchoTrainLength])
    for i in range(len(ph_list)):
        NE, NS = np.unravel_index(i, [len(ph_list) / EchoTrainLength, EchoTrainLength])
        ph[:, :, NS, NE] = np.transpose(np.single(pydicom.dcmread(os.path.join(path_ph, ph_list[i])).pixel_array))
        ph[:, :, NS, NE] = ph[:, :, NS, NE] / 4095 * 2 * np.pi - np.pi

    # % interpolation into isotropic
    # if interp
    if interp:
        # img = mag.*exp(1j*ph);
        img = mag * np.exp(1j * ph)
        # k = fftshift(fftshift(fftshift(fft(fft(fft(img,[],1),[],2),[],3),1),2),3);
        k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img, axes=(0, 1, 2)), axes=(0, 1, 2)))
        # % find the finest resolution
        # minvox = min(vox);
        minvox = np.min(vox)
        # % update matrix size
        # pad_size =  round((vox.*imsize(1:3)/minvox - imsize(1:3))/2);
        pad_size = np.round((vox * imsize[0:3] / minvox - imsize[0:3]) / 2)
        # k = padarray(k, pad_size);
        k = padarray(k, pad_size)
        # img = ifft(ifft(ifft(ifftshift(ifftshift(ifftshift(k,1),2),3),[],1),[],2),[],3);
        img = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(k, axes=(0, 1, 2)), axes=(0, 1, 2)), axes=(0, 1, 2))
        # clear k;
        del k
        # imsize_old = imsize;
        imsize_old = imsize
        # imsize = size(img);
        imsize = img.shape
        # vox = imsize_old(1:3).*vox./imsize(1:3);
        vox = imsize_old[0:3] * vox / imsize[0:3]
        # mag = abs(img);
        mag = np.abs(img)
        # ph = angle(img);
        ph = np.angle(img)
        # end

    # % define output directories
    # path_qsm = [path_out '/QSM_R2S_PRISMA'];
    path_qsm = os.path.join(path_out, 'QSM_R2S_PRISMA')
    # mkdir(path_qsm);
    os.mkdir(path_qsm)
    # init_dir = pwd;
    init_dir = os.getcwd()
    # cd(path_qsm);  # 还是用cd，但是之前写的os.path.join就不需要动了，因为本就是绝对路径
    os.chdir(path_qsm)

    # % save magnitude and raw phase niftis for each echo
    # mkdir('src')
    os.mkdir(os.path.join(path_qsm, 'src'))
    # for echo = 1:imsize(4)
    #     nii = make_nii(mag(:,:,:,echo),vox);
    #     save_nii(nii,['src/mag' num2str(echo) '.nii']);
    #     nii = make_nii(ph(:,:,:,echo),vox);
    #     save_nii(nii,['src/ph' num2str(echo) '.nii']);
    # end
    for echo in range(imsize[3]):
        # 改用nib的库来保存nii
        # make_save_nii_engine(mag[:, :, :, echo], f"echo{echo}", vox, f"sec/mag{echo}.nii")
        nii = nib.Nifti1Image(mag[:, :, :, echo], np.eye(4))
        nib.save(nii, os.path.join(path_qsm, 'src', f"mag{echo}.nii"))
        # make_save_nii_engine(ph[:, :, :, echo], f"echo{echo}", vox, f"sec/ph{echo}.nii")
        nii = nib.Nifti1Image(ph[:, :, :, echo], np.eye(4))
        nib.save(nii, os.path.join(path_qsm, 'src', f"ph{echo}.nii"))

    # % brain extraction
    # % generate mask from magnitude of the 1th echo
    # disp('--> extract brain volume and generate mask ...');
    print('--> extract brain volume and generate mask ...')
    # setenv('bet_thr',num2str(bet_thr));
    os.environ['bet_thr'] = str(bet_thr)
    # setenv('bet_smooth',num2str(bet_smooth));
    os.environ['bet_smooth'] = str(bet_smooth)
    # TODO 这里的 unix真的没办法了, 先翻译，跑还是linux跑
    # [~,~] = unix('rm BET*');
    subprocess.run('rm BET*', shell=True)
    # unix('bet2 src/mag1.nii BET -f ${bet_thr} -m -w ${bet_smooth}');
    subprocess.run(f'bet2 {os.path.join(path_qsm, "src/mag1.nii")} BET -f ${bet_thr} -m -w ${bet_smooth}', shell=True)
    # unix('gunzip -f BET.nii.gz');
    subprocess.run('gunzip -f BET.nii.gz', shell=True)
    # unix('gunzip -f BET_mask.nii.gz');
    subprocess.run('gunzip -f BET_mask.nii.gz', shell=True)
    # nii = load_nii('BET_mask.nii'); TODO 直接采用nib加载了nii文件，但不确定加载的数据格式是否要调整。
    nii = nib.load(os.path.join(path_qsm, 'BET_mask.nii'))
    # mask = double(nii.img);
    mask = np.double(nii.img)

    # % phase offset correction
    # % if unipolar
    # if strcmpi('unipolar',readout)
    #     ph_corr = geme_cmb(mag.*exp(1j*ph),vox,TE,mask);
    if readout.lower() == 'unipolar':
        ph_corr = geme_cmb(mag * np.exp(1j * ph), vox, TE, mask)
    # % if bipolar
    # elseif strcmpi('bipolar',readout)
    elif readout.lower() == 'bipolar':
        # ph_corr = zeros(imsize);
        ph_corr = np.zeros(imsize)
        #   ph_corr(:,:,:,1:2:end) = geme_cmb(mag(:,:,:,1:2:end).*exp(1j*ph(:,:,:,1:2:end)),vox,TE(1:2:end),mask);
        ph_corr[:, :, :, 0:2:-1] = geme_cmb(mag[:, :, :, 0:2:-1] * np.exp(1j * ph[:, :, :, 0:2:-1]),
                                            vox, TE[0:2:-1], mask)
        #   ph_corr(:,:,:,2:2:end) = geme_cmb(mag(:,:,:,2:2:end).*exp(1j*ph(:,:,:,2:2:end)),vox,TE(2:2:end),mask);
        ph_corr[:, :, :, 1:2:-1] = geme_cmb(mag[:, :, :, 1:2:-1] * np.exp(1j * ph[:, :, :, 1:2:-1]),
                                            vox, TE[1:2:-1], mask)
    # else
    else:
        #   error('is the sequence unipolar or bipolar readout?')
        raise ValueError('is the sequence unipolar or bipolar readout?')

    # % save offset corrected phase niftis
    # for echo = 1:imsize(4)
    #     nii = make_nii(ph_corr(:,:,:,echo),vox);
    #     save_nii(nii,['src/ph_corr' num2str(echo) '.nii']);
    # end
    for echo in range(imsize[3]):
        # make_save_nii_engine(ph_corr[:, :, :, echo], f"echo{echo}", vox, f"sec/ph_corr{echo}.nii")
        nii = nib.Nifti1Image(ph_corr[:, :, :, echo], np.eye(4))
        nib.save(nii, os.path.join(path_qsm, 'src', f"ph_corr{echo}.nii"))

    # % unwrap phase from each echo
    # if strcmpi('prelude',ph_unwrap)
    if ph_unwrap.lower() == "prelude":
        # disp('--> unwrap aliasing phase for all TEs using prelude...');
        print('--> unwrap aliasing phase for all TEs using prelude...')
        # setenv('echo_num',num2str(imsize(4)));
        os.environ['echo_num'] = str(imsize[3])
        # bash_command = sprintf(['for ph in src/ph_corr[1-$echo_num].nii\n' ...
        # 'do\n' ...
        # 'base=`basename $ph`;\n' ...
        # 'dir=`dirname $ph`;\n' ...
        # 'mag=$dir/"mag"${base:7};\n' ...
        # 'unph="unph"${base:7};\n' ...
        # 'prelude -a $mag -p $ph -u $unph -m BET_mask.nii -n 12&\n' ...
        # 'done\n' ...
        # 'wait\n' ...
        # 'gunzip -f unph*.gz\n']);
        # 在 MATLAB 中，上述代码的作用是通过使用 prelude 程序对一组 NIfTI 格式的磁共振相位图像进行处理。
        # 具体来说，它通过循环处理 src/ph_corr[1-$echo_num].nii 这些文件，并执行以下操作：
        #
        # 从路径中获取文件名和目录名。
        # 构建相应的磁共振幅值图像 mag 和未包裹相位图像 unph 的文件名。
        # 使用 prelude 程序处理 mag 和 ph，并生成 unph 文件。
        # 在后台运行 prelude 命令，以便能够继续执行其他命令。
        # 等待所有后台任务完成。
        # 解压缩所有生成的 unph 文件。

        bash_command = '''
        for ph in src/ph_corr[1-$echo_num].nii
        do
            base=$(basename $ph);
            dir=$(dirname $ph);
            mag=$dir/"mag"${base:7};
            unph="unph"${base:7};
            prelude -a $mag -p $ph -u $unph -m BET_mask.nii -n 12&
        done
        wait
        gunzip -f unph*.gz
        '''

        subprocess.run(bash_command, shell=True)

        # unph = zeros(imsize);
        unph = np.zeros(imsize)
        # for echo = 1:imsize(4)
        #     nii = load_nii(['unph' num2str(echo) '.nii']);
        #     unph(:,:,:,echo) = double(nii.img);
        # end
        for echo in range(imsize[3]):
            # TODO 依旧使用nib库加载nii文件
            nii = nib.load(f'unph{echo}.nii')
            unph[:, :, :, echo] = np.double(nii.img)

    # elseif strcmpi('bestpath',ph_unwrap)
    elif ph_unwrap.lower() == "bestpath":
        # % unwrap the phase using best path
        # disp('--> unwrap aliasing phase using bestpath...');
        print('--> unwrap aliasing phase using bestpath...')
        # mask_unwrp = uint8(abs(mask)*255);
        mask_unwrp = np.uint8(np.abs(mask) * 255)
        # fid = fopen('mask_unwrp.dat','w');
        # fwrite(fid,mask_unwrp,'uchar');
        # fclose(fid);
        with open(os.path.join(path_mag, 'mask_unwrp.dat'), 'wb') as fid:
            fid.write(mask_unwrp)

        # [pathstr, ~, ~] = fileparts(which('3DSRNCP.m')); determine the path of 3DSRNCP.m
        abs_path = os.path.dirname(os.path.abspath(__file__)).split("Libs")[0]
        pathstr = abs_path + "Libs\\phase_unwrap\\"
        # setenv('pathstr',pathstr);
        # setenv('nv',num2str(imsize(1)));
        # setenv('np',num2str(imsize(2)));
        # setenv('ns',num2str(imsize(3)));
        os.environ['pathstr'] = pathstr
        os.environ['nv'] = str(imsize[0])
        os.environ['np'] = str(imsize[1])
        os.environ['ns'] = str(imsize[2])

        # unph = zeros(imsize);
        unph = np.zeros(imsize)
        # for echo_num = 1:imsize(4)
        for echo_num in range(imsize[3]):
            # setenv('echo_num',num2str(echo_num));
            os.environ['echo_num'] = str(echo_num)
            # fid = fopen(['wrapped_phase' num2str(echo_num) '.dat'],'w');
            # fwrite(fid,ph_corr(:,:,:,echo_num),'float');
            # fclose(fid);
            # 因为是dat文件，所以还是用write的形式写入。
            with open(os.path.join(path_mag, f'wrapped_phase{echo_num}.dat'), 'wb') as fid:
                # 用numpy的tofile方法写入
                ph_corr[:, :, :, echo_num].tofile(fid)
                # fid.write(ph_corr[:, :, :, echo_num])

            # bash_script = ['${pathstr}/3DSRNCP wrapped_phase${echo_num}.dat mask_unwrp.dat
            #                   unwrapped_phase${echo_num}.dat $nv $np $ns reliability${echo_num}.dat'];
            bash_script = (f'{pathstr}3DSRNCP wrapped_phase{echo_num}.mat mask_unwrp.mat '
                           f'unwrapped_phase{echo_num}.mat $nv $np $ns reliability{echo_num}.mat')
            # unix(bash_script) ;
            subprocess.run(bash_script, shell=True)
            # fid = fopen(['unwrapped_phase' num2str(echo_num) '.dat'],'r');
            # tmp = fread(fid,'float');
            tmp = np.fromfile(os.path.join(path_mag, f'unwrapped_phase{echo_num}.mat'), dtype=np.float32)
            # % tmp = tmp - tmp(1);
            # unph(:,:,:,echo_num) = reshape(tmp - round(mean(tmp(mask==1))/(2*pi))*2*pi ,imsize(1:3)).*mask;
            unph[:, :, :, echo_num] = np.reshape(
                tmp - np.round(np.mean(tmp[mask == 1])) / ((2 * np.pi) * 2 * np.pi, imsize[0:3])
            ) * mask

            # fid = fopen(['reliability' num2str(echo_num) '.dat'],'r');
            # reliability_raw = fread(fid,'float');
            # reliability_raw = reshape(reliability_raw,imsize(1:3));
            # fclose(fid);
            reliability_raw = np.fromfile(os.path.join(path_mag, f'reliability{echo_num}.mat'), dtype=np.float32)
            reliability_raw = np.reshape(reliability_raw, imsize[0:3])

            # nii = make_nii(reliability_raw.*mask,vox);
            # save_nii(nii,['reliability_raw' num2str(echo_num) '.nii']);
            # make_save_nii_engine(reliability_raw * mask, f"reliability_raw{echo_num}", vox,
            #                      os.path.join(path_mag, f'reliability_raw{echo_num}.nii'))
            nii = nib.Nifti1Image(reliability_raw * mask, np.eye(4))
            nib.save(nii, os.path.join(path_mag, f'reliability_raw{echo_num}.nii'))
        # end
        #
        # nii = make_nii(unph,vox);
        # save_nii(nii,'unph_bestpath.nii');
        make_save_nii_engine(unph, 'unph_bestpath', vox, os.path.join(path_mag, 'unph_bestpath.nii'))
    #
    # else
    else:
        # error('what unwrapping methods to use? prelude or bestpath?')
        raise ValueError('what unwrapping methods to use? prelude or bestpath?')
    # end

    # % check and correct for 2pi jump between echoes
    # disp('--> correct for potential 2pi jumps between TEs ...')
    print('--> correct for potential 2pi jumps between TEs ...')

    # nii = load_nii('unph_diff.nii');
    # unph_diff = double(nii.img);
    unph_diff = nib.load(os.path.join(path_mag, 'unph_diff.nii')).get_fdata().astype(np.float64)

    # for echo = 2:imsize(4)
    # meandiff = unph(:,:,:,echo)-unph(:,:,:,1)-double(echo-1)*unph_diff;
    # meandiff = meandiff(mask==1);
    # meandiff = mean(meandiff(:));
    # njump = round(meandiff/(2*pi));
    # disp(['    ' num2str(njump) ' 2pi jumps for TE' num2str(echo)]);
    # unph(:,:,:,echo) = unph(:,:,:,echo) - njump*2*pi;
    # unph(:,:,:,echo) = unph(:,:,:,echo).*mask;
    # end
    for echo in range(1, imsize[3]):
        meandiff = unph[:, :, :, echo] - unph[:, :, :, 0] - (echo - 1) * unph_diff
        meandiff = meandiff[mask == 1]
        meandiff = np.mean(meandiff)
        njump = np.round(meandiff / (2 * np.pi))
        print(f'    {njump} 2pi jumps for TE{echo}')
        unph[:, :, :, echo] = unph[:, :, :, echo] - njump * 2 * np.pi
        unph[:, :, :, echo] = unph[:, :, :, echo] * mask

    # % fit phase images with echo times
    # disp('--> magnitude weighted LS fit of phase to TE ...');
    # [tfs, fit_residual] = echofit(unph,mag,TE,0);
    print('--> magnitude weighted LS fit of phase to TE ...')
    # TODO echofit需要翻译
    tfs, fit_residual = echofit(unph, mag, TE, 0)

    # % extra filtering according to fitting residuals
    # if r_mask
    #     % generate reliability map
    #     fit_residual_blur = smooth3(fit_residual,'box',round(1./vox)*2+1);
    #     nii = make_nii(fit_residual_blur,vox);
    #     save_nii(nii,'fit_residual_blur.nii');
    #     R = ones(size(fit_residual_blur));
    #     R(fit_residual_blur >= fit_thr) = 0;
    if r_mask:
        fit_residual_blur = ndimage.uniform_filter(fit_residual, size=np.round(1. / vox) * 2 + 1)
        nii = nib.Nifti1Image(fit_residual_blur, np.eye(4))
        nib.save(nii, os.path.join(path_qsm, 'fit_residual_blur.nii'))
        R = np.ones(fit_residual_blur.shape)
        R[fit_residual_blur >= fit_thr] = 0
    # else
    #     R = 1;
    # end
    else:
        R = 1

    # % normalize to main field
    # % ph = gamma*dB*TE
    # % dB/B = ph/(gamma*TE*B0)
    # % units: TE s, gamma 2.675e8 rad/(sT), B0 4.7T
    # tfs = tfs/(2.675e8*dicom_info.MagneticFieldStrength)*1e6; % unit ppm
    tfs = tfs / (2.675e8 * dicom_info.MagneticFieldStrength) * 1e6  # unit ppm
    # nii = make_nii(tfs,vox);
    # save_nii(nii,'tfs.nii');
    nii = nib.Nifti1Image(tfs, np.eye(4))
    nib.save(nii, os.path.join(path_qsm, 'tfs.nii'))

    # % LN-QSM
    # if sum(strcmpi('lnqsm',bkg_rm))
    if 'lnqsm' in bkg_rm:
        # mkdir LN-QSM
        os.mkdir("LN-QSM")

        # maskR = mask.*R;
        # iMag = sqrt(sum(mag.^2,4));
        # Res_wt = iMag.*maskR;
        # Res_wt = Res_wt/sum(Res_wt(:))*sum(maskR(:));
        maskR = mask * R
        iMag = np.sqrt(np.sum(mag ** 2, axis=3))
        Res_wt = iMag * maskR
        Res_wt = Res_wt / np.sum(Res_wt) * np.sum(maskR)

        # % mask_erosion
        # r = 1;
        # [X,Y,Z] = ndgrid(-r:r,-r:r,-r:r);
        # h = (X.^2/r^2 + Y.^2/r^2 + Z.^2/r^2 <= 1);
        # ker = h/sum(h(:));
        # imsize = size(mask);
        # mask_tmp = convn(mask,ker,'same');
        # mask_ero1 = zeros(imsize);
        # mask_ero1(mask_tmp > 0.999999) = 1; % no error tolerance
        r = 1
        X, Y, Z = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), np.arange(-r, r + 1), indexing='ij')
        h = (X ** 2 / r ** 2 + Y ** 2 / r ** 2 + Z ** 2 / r ** 2 <= 1)
        ker = h / np.sum(h)
        imsize = mask.shape
        mask_tmp = np.convolve(mask, ker, mode='same')
        mask_ero1 = np.zeros(imsize)
        mask_ero1[mask_tmp > 0.999999] = 1  # no error tolerance

        # r = 2;
        # [X,Y,Z] = ndgrid(-r:r,-r:r,-r:r);
        # h = (X.^2/r^2 + Y.^2/r^2 + Z.^2/r^2 <= 1);
        # ker = h/sum(h(:));
        # imsize = size(mask);
        # mask_tmp = convn(mask,ker,'same');
        # mask_ero2 = zeros(imsize);
        # mask_ero2(mask_tmp > 0.999999) = 1; % no error tolerance
        r = 2
        X, Y, Z = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), np.arange(-r, r + 1), indexing='ij')
        h = (X ** 2 / r ** 2 + Y ** 2 / r ** 2 + Z ** 2 / r ** 2 <= 1)
        ker = h / np.sum(h)
        imsize = mask.shape
        mask_tmp = np.convolve(mask, ker, mode='same')
        mask_ero2 = np.zeros(imsize)
        mask_ero2[mask_tmp > 0.999999] = 1  # no error tolerance

        # r = 3;
        # [X,Y,Z] = ndgrid(-r:r,-r:r,-r:r);
        # h = (X.^2/r^2 + Y.^2/r^2 + Z.^2/r^2 <= 1);
        # ker = h/sum(h(:));
        # imsize = size(mask);
        # mask_tmp = convn(mask,ker,'same');
        # mask_ero3 = zeros(imsize);
        # mask_ero3(mask_tmp > 0.999999) = 1; % no error tolerance
        r = 3
        X, Y, Z = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), np.arange(-r, r + 1), indexing='ij')
        h = (X ** 2 / r ** 2 + Y ** 2 / r ** 2 + Z ** 2 / r ** 2 <= 1)
        ker = h / np.sum(h)
        imsize = mask.shape
        mask_tmp = np.convolve(mask, ker, mode='same')
        mask_ero3 = np.zeros(imsize)
        mask_ero3[mask_tmp > 0.999999] = 1  # no error tolerance

        # % (1) no erosion, keep the full brain
        # P = maskR + 30*(1 - maskR);
        # chi_ero0_500 = tikhonov_qsm(tfs, Res_wt.*maskR, 1, maskR, maskR, 0, 1e-4, 0.001, 0, vox, P, z_prjs, 500);
        # nii = make_nii(chi_ero0_500.*maskR,vox);
        # save_nii(nii,'LN-QSM/chi_ero0_tik_1e-3_tv_1e-4_500.nii');
        P = maskR + 30 * (1 - maskR)
        chi_ero0_500 = tikhonov_qsm(tfs, Res_wt * maskR, 1, maskR, maskR, 0, 1e-4, 0.001, 0, vox, P, z_prjs, 500)
        nii = nib.Nifti1Image(chi_ero0_500 * maskR, np.eye(4))
        nib.save(nii, '\\LN-QSM\\chi_ero0_tik_1e-3_tv_1e-4_500.nii.gz')

        # % (2) erode 1 voxel from brain edge
        # P = mask_ero1 + 30*(1 - mask_ero1);
        # chi_ero1_500 = tikhonov_qsm(tfs, Res_wt.*mask_ero1, 1, mask_ero1,
        #                             mask_ero1, 0, 1e-4, 0.001, 0, vox, P, z_prjs, 500);
        # nii = make_nii(chi_ero1_500.*mask_ero1,vox);
        # save_nii(nii,'LN-QSM/chi_ero1_tik_1e-3_tv_1e-4_500.nii');
        P = mask_ero1 + 30 * (1 - mask_ero1)
        chi_ero1_500 = tikhonov_qsm(tfs, Res_wt * mask_ero1, 1, mask_ero1, mask_ero1, 0, 1e-4, 0.001, 0, vox, P, z_prjs,
                                    500)
        nii = nib.Nifti1Image(chi_ero1_500 * mask_ero1, np.eye(4))
        nib.save(nii, '\\LN-QSM\\chi_ero1_tik_1e-3_tv_1e-4_500.nii.gz')

        # % (3) erode 2 voxel from brain edge
        # P = mask_ero2 + 30*(1 - mask_ero2);
        # chi_ero2_500 = tikhonov_qsm(tfs, Res_wt.*mask_ero2, 1,
        #                             mask_ero2, mask_ero2, 0, 1e-4, 0.001, 0, vox, P, z_prjs, 500);
        # nii = make_nii(chi_ero2_500.*mask_ero2,vox);
        # save_nii(nii,'LN-QSM/chi_ero2_tik_1e-3_tv_1e-4_500.nii');
        P = mask_ero2 + 30 * (1 - mask_ero2)
        chi_ero2_500 = tikhonov_qsm(tfs, Res_wt * mask_ero2, 1, mask_ero2, mask_ero2, 0, 1e-4, 0.001, 0, vox, P, z_prjs,
                                    500)
        nii = nib.Nifti1Image(chi_ero2_500 * mask_ero2, np.eye(4))
        nib.save(nii, '\\LN-QSM\\chi_ero2_tik_1e-3_tv_1e-4_500.nii.gz')

        # % (4) erode 3 voxel from brain edge
        # P = mask_ero3 + 30*(1 - mask_ero3);
        # chi_ero3_500 = tikhonov_qsm(tfs, Res_wt.*mask_ero3, 1,
        #                             mask_ero3, mask_ero3, 0, 1e-4, 0.001, 0, vox, P, z_prjs, 500);
        # nii = make_nii(chi_ero3_500.*mask_ero3,vox);
        # save_nii(nii,'LN-QSM/chi_ero3_tik_1e-3_tv_1e-4_500.nii');
        P = mask_ero3 + 30 * (1 - mask_ero3)
        chi_ero3_500 = tikhonov_qsm(tfs, Res_wt * mask_ero3, 1, mask_ero3, mask_ero3, 0, 1e-4, 0.001, 0, vox, P, z_prjs,
                                    500)
        nii = nib.Nifti1Image(chi_ero3_500 * mask_ero3, np.eye(4))

    # % background field removal
    # % PDF
    # if sum(strcmpi('pdf',bkg_rm))
    if 'pdf' in bkg_rm:
        # disp('--> PDF to remove background field ...');
        # [lfs_pdf,mask_pdf] = projectionontodipolefields(tfs,mask.*R,vox,smv_rad,mag(:,:,:,end),z_prjs);
        # % 3D 2nd order polyfit to remove any residual background
        # lfs_pdf= lfs_pdf - poly3d(lfs_pdf,mask_pdf);
        print('--> PDF to remove background field ...')
        lfs_pdf, mask_pdf = projectionontodipolefields(tfs, mask * R, vox, smv_rad, mag[..., -1], z_prjs)
        # % 3D 2nd order polyfit to remove any residual background
        lfs_pdf = lfs_pdf - poly3d(lfs_pdf, mask_pdf)

        # % save nifti
        # mkdir('PDF');
        os.mkdir('PDF')
        # nii = make_nii(lfs_pdf,vox);
        # save_nii(nii,'PDF/lfs_pdf.nii');
        nii = nib.Nifti1Image(lfs_pdf, np.eye(4))
        nib.save(nii, '\\PDF\\lfs_pdf.nii.gz')

        # % inversion of susceptibility
        # disp('--> TV susceptibility inversion on PDF...');
        # sus_pdf = tvdi(lfs_pdf,mask_pdf,vox,tv_reg,mag(:,:,:,end),z_prjs,inv_num);
        print('--> TV susceptibility inversion on PDF...')
        sus_pdf = tvdi(lfs_pdf, mask_pdf, vox, tv_reg, mag[..., -1], z_prjs, inv_num)

        # % save nifti
        # nii = make_nii(sus_pdf.*mask_pdf,vox);
        # save_nii(nii,'PDF/sus_pdf.nii');
        nii = nib.Nifti1Image(sus_pdf * mask_pdf, np.eye(4))
        nib.save(nii, '\\PDF\\sus_pdf.nii.gz')

    # % SHARP (t_svd: truncation threthold for t_svd)
    # if sum(strcmpi('sharp',bkg_rm))
    if 'sharp' in bkg_rm:
        # disp('--> SHARP to remove background field ...');
        # [lfs_sharp, mask_sharp] = sharp(tfs,mask.*R,vox,smv_rad,t_svd);
        # % % 3D 2nd order polyfit to remove any residual background
        # % lfs_sharp= poly3d(lfs_sharp,mask_sharp);
        print('--> SHARP to remove background field ...')
        lfs_sharp, mask_sharp = sharp(tfs, mask * R, vox, smv_rad, t_svd)

        # % save nifti
        # mkdir('SHARP');
        # nii = make_nii(lfs_sharp,vox);
        # save_nii(nii,'SHARP/lfs_sharp.nii');
        os.mkdir('SHARP')
        nii = nib.Nifti1Image(lfs_sharp, np.eye(4))
        nib.save(nii, '\\SHARP\\lfs_sharp.nii.gz')

        # % inversion of susceptibility
        # disp('--> TV susceptibility inversion on SHARP...');
        # sus_sharp = tvdi(lfs_sharp,mask_sharp,vox,tv_reg,mag(:,:,:,end),z_prjs,inv_num);
        print('--> TV susceptibility inversion on SHARP...')
        sus_sharp = tvdi(lfs_sharp, mask_sharp, vox, tv_reg, mag[..., -1], z_prjs, inv_num)

        # % save nifti
        # nii = make_nii(sus_sharp.*mask_sharp,vox);
        # save_nii(nii,'SHARP/sus_sharp.nii');
        nii = nib.Nifti1Image(sus_sharp * mask_sharp, np.eye(4))
        nib.save(nii, '\\SHARP\\sus_sharp.nii.gz')

    # % RE-SHARP (tik_reg: Tikhonov regularization parameter)
    # if sum(strcmpi('resharp',bkg_rm))
    if 'resharp' in bkg_rm:
        # disp('--> RESHARP to remove background field ...');
        # [lfs_resharp, mask_resharp] = resharp(tfs,mask.*R,vox,smv_rad,tik_reg,cgs_num);
        # % % 3D 2nd order polyfit to remove any residual background
        # % lfs_resharp= poly3d(lfs_resharp,mask_resharp);
        print('--> RESHARP to remove background field ...')
        lfs_resharp, mask_resharp = resharp(tfs, mask * R, vox, smv_rad, tik_reg, cgs_num)

        # % save nifti
        # mkdir('RESHARP');
        # nii = make_nii(lfs_resharp,vox);
        # save_nii(nii,'RESHARP/lfs_resharp.nii');
        os.mkdir('RESHARP')
        nii = nib.Nifti1Image(lfs_resharp, np.eye(4))
        nib.save(nii, '\\RESHARP\\lfs_resharp.nii.gz')

        # % inversion of susceptibility
        # disp('--> TV susceptibility inversion on RESHARP...');
        # % iLSQR
        # chi_iLSQR = QSM_iLSQR(lfs_resharp*(2.675e8*dicom_info.MagneticFieldStrength)/1e6,mask_resharp,'H',z_prjs,'voxelsize',vox,'niter',50,'TE',1000,'B0',dicom_info.MagneticFieldStrength);
        # nii = make_nii(chi_iLSQR,vox);
        # save_nii(nii,['RESHARP/chi_iLSQR_smvrad' num2str(smv_rad) '.nii']);
        print('--> TV susceptibility inversion on RESHARP...')
        # TODO 找不到QSM_iLSQR函数
        chi_iLSQR = QSM_iLSQR(lfs_resharp * (2.675e8 * dicom_info.MagneticFieldStrength) / 1e6, mask_resharp, 'H', z_prjs, 'voxelsize', vox, 'niter', 50, 'TE', 1000, 'B0', dicom_info.MagneticFieldStrength)
        nii = nib.Nifti1Image(chi_iLSQR, np.eye(4))
        nib.save(nii, '\\RESHARP\\chi_iLSQR_smvrad' + str(smv_rad) + '.nii')

        # % % MEDI 该部分被大量的注释掉了，遂不保留
        # % %%%%% normalize signal intensity by noise to get SNR %%%
        # % %%%% Generate the Magnitude image %%%%

        # % TVDI method
        # sus_resharp = tvdi(lfs_resharp,mask_resharp,vox,tv_reg,mag(:,:,:,end),z_prjs,inv_num);
        # nii = make_nii(sus_resharp.*mask_resharp,vox);
        # save_nii(nii,'RESHARP/sus_resharp.nii');
        sus_resharp = tvdi(lfs_resharp, mask_resharp, vox, tv_reg, mag[..., -1], z_prjs, inv_num)
        nii = nib.Nifti1Image(sus_resharp * mask_resharp, np.eye(4))
        nib.save(nii, '\\RESHARP\\sus_resharp.nii')

    # % V-SHARP
    # if sum(strcmpi('vsharp',bkg_rm))
    if 'vsharp' in bkg_rm:
        # disp('--> V-SHARP to remove background field ...');
        # smvsize = 12;
        # [lfs_vsharp, mask_vsharp] = V_SHARP(tfs ,single(mask.*R),'smvsize',smvsize,'voxelsize',vox);
        print('--> V-SHARP to remove background field ...')
        smvsize = 12
        # TODO V_SHARP函数找不到
        lfs_vsharp, mask_vsharp = V_SHARP(tfs, mask * R, 'smvsize', smvsize, 'voxelsize', vox)

        # % save nifti
        # mkdir('VSHARP');
        # nii = make_nii(lfs_vsharp,vox);
        # save_nii(nii,'VSHARP/lfs_vsharp.nii');
        os.mkdir('VSHARP')
        nii = nib.Nifti1Image(lfs_vsharp, np.eye(4))
        nib.save(nii, '\\VSHARP\\lfs_vsharp.nii.gz')
        # % inversion of susceptibility
        # disp('--> TV susceptibility inversion on RESHARP...');
        # sus_vsharp = tvdi(lfs_vsharp,mask_vsharp,vox,tv_reg,mag(:,:,:,end),z_prjs,inv_num);
        print('--> TV susceptibility inversion on RESHARP...')
        sus_vsharp = tvdi(lfs_vsharp, mask_vsharp, vox, tv_reg, mag[..., -1], z_prjs, inv_num)

        # % save nifti
        # nii = make_nii(sus_vsharp.*mask_vsharp,vox);
        # save_nii(nii,'VSHARP/sus_vsharp.nii');
        nii = nib.Nifti1Image(sus_vsharp * mask_vsharp, np.eye(4))
        nib.save(nii, '\\VSHARP\\sus_vsharp.nii')

    # % E-SHARP (SHARP edge extension)
    # if sum(strcmpi('esharp',bkg_rm))
    if 'esharp' in bkg_rm:
        # disp('--> E-SHARP to remove background field ...');
        # Parameters.voxelSize             = vox; % in mm
        # Parameters.resharpRegularization = tik_reg ;
        # Parameters.resharpKernelRadius   = smv_rad ; % in mm
        # Parameters.radius                = [ 10 10 5 ] ;
        print('--> E-SHARP to remove background field ...')
        Parameters = Param()
        Parameters.voxelSize = vox
        Parameters.resharpRegularization = tik_reg
        Parameters.resharpKernelRadius = smv_rad
        Parameters.radius = [10, 10, 5]

        # % pad matrix size to even number
        # pad_size = mod(size(tfs),2);
        # tfs = tfs.*mask.*R;
        # tfs = padarray(tfs, pad_size, 'post');
        pad_size = np.mod(tfs.shape, 2)
        tfs = tfs * mask * R
        tfs = np.pad(tfs, ((0, pad_size[0]), (0, pad_size[1]), (0, pad_size[2])), 'constant', constant_values=0)

        # % taking off additional 1 voxels from edge - not sure the outermost
        # % phase data included in the original mask is reliable.
        # mask_shaved = shaver( ( tfs ~= 0 ), 1 ) ; % 1 voxel taken off
        # totalField  = mask_shaved .* tfs ;
        mask_shaved = shaver((tfs != 0), 1)
        totalField = mask_shaved * tfs

        # % resharp
        # [reducedLocalField, maskReduced] = resharp(totalField,
        #                                    double(mask_shaved),
        #                                    Parameters.voxelSize,
        #                                    Parameters.resharpKernelRadius,
        #                                    Parameters.resharpRegularization ) ;
        reducedLocalField, maskReduced = resharp(totalField,
                                                 mask_shaved,
                                                 Parameters.voxelSize,
                                                 Parameters.resharpKernelRadius,
                                                 Parameters.resharpRegularization)

        # % extrapolation ~ esharp
        # reducedBackgroundField = maskReduced .* ( totalField - reducedLocalField) ;
        reducedBackgroundField = maskReduced * (totalField - reducedLocalField)

        # extendedBackgroundField = extendharmonicfield(reducedBackgroundField, mask, maskReduced, Parameters);
        extendedBackgroundField = extendharmonicfield(reducedBackgroundField, mask, maskReduced, Parameters)

        # backgroundField = extendedBackgroundField + reducedBackgroundField ;
        # localField      = totalField - backgroundField ;
        extendedBackgroundField = extendedBackgroundField + reducedBackgroundField
        localField = totalField - extendedBackgroundField

        # lfs_esharp      = localField(1+pad_size(1):end,1+pad_size(2):end,1+pad_size(3):end);
        # mask_esharp     = mask_shaved(1+pad_size(1):end,1+pad_size(2):end,1+pad_size(3):end);
        lfs_esharp = localField[pad_size[0]:, pad_size[1]:, pad_size[2]:]
        mask_esharp = mask_shaved[pad_size[0]:, pad_size[1]:, pad_size[2]:]

        # % % 3D 2nd order polyfit to remove any residual background
        # % lfs_esharp = poly3d(lfs_esharp,mask_esharp);

        # % save nifti
        # mkdir('ESHARP');
        # nii = make_nii(lfs_esharp,vox);
        # save_nii(nii,'ESHARP/lfs_esharp.nii');
        os.mkdir('ESHARP')
        nii = nib.Nifti1Image(lfs_esharp, np.eye(4))
        nib.save(nii, '\\ESHARP\\lfs_esharp.nii.gz')

        # % inversion of susceptibility
        # disp('--> TV susceptibility inversion on ESHARP...');
        # sus_esharp = tvdi(lfs_esharp,mask_esharp,vox,tv_reg,mag(:,:,:,end),z_prjs,inv_num);
        print('--> TV susceptibility inversion on ESHARP...')
        sus_esharp = tvdi(lfs_esharp, mask_esharp, vox, tv_reg, mag[..., -1], z_prjs, inv_num)

        # % save nifti
        # nii = make_nii(sus_esharp.*mask_esharp,vox);
        # save_nii(nii,'ESHARP/sus_esharp.nii');
        nii = nib.Nifti1Image(sus_esharp * mask_esharp, np.eye(4))
        nib.save(nii, '\\ESHARP\\sus_esharp.nii')
    # end

    # % LBV
    # if sum(strcmpi('lbv',bkg_rm))
    if 'lbv' in bkg_rm:
        # disp('--> LBV to remove background field ...');
        # lfs_lbv = LBV(tfs,mask.*R,imsize(1:3),vox,lbv_tol,lbv_peel); % strip 2 layers
        # mask_lbv = ones(imsize(1:3));
        # mask_lbv(lfs_lbv==0) = 0;
        # % 3D 2nd order polyfit to remove any residual background
        # lfs_lbv = lfs_lbv - poly3d(lfs_lbv,mask_lbv);
        print('--> LBV to remove background field ...')
        # TODO LBV函数找不到
        lfs_lbv = LBV(tfs, mask * R, imsize[:3], vox, lbv_tol, lbv_peel)  # strip 2 layers
        mask_lbv = np.ones(imsize[:3])
        mask_lbv[lfs_lbv == 0] = 0
        # % 3D 2nd order polyfit to remove any residual background
        lfs_lbv = lfs_lbv - poly3d(lfs_lbv, mask_lbv)

        # % save nifti
        # mkdir('LBV');
        # nii = make_nii(lfs_lbv,vox);
        # save_nii(nii,'LBV/lfs_lbv.nii');
        os.mkdir('LBV')
        nii = nib.Nifti1Image(lfs_lbv, np.eye(4))
        nib.save(nii, '\\LBV\\lfs_lbv.nii')

        # % inversion of susceptibility
        # disp('--> TV susceptibility inversion on LBV...');
        print('--> TV susceptibility inversion on LBV...')
        # % iLSQR
        # chi_iLSQR = QSM_iLSQR(lfs_lbv*(2.675e8*dicom_info.MagneticFieldStrength)/1e6,mask_lbv,'H',z_prjs,'voxelsize',vox,'niter',50,'TE',1000,'B0',dicom_info.MagneticFieldStrength);
        # nii = make_nii(chi_iLSQR,vox);
        # save_nii(nii,['LBV/chi_iLSQR_smvrad' num2str(smv_rad) '.nii']);
        chi_iLSQR = QSM_iLSQR(lfs_lbv * (2.675e8 * dicom_info.MagneticFieldStrength) / 1e6, mask_lbv, 'H', z_prjs, 'voxelsize', vox, 'niter', 50, 'TE', 1000, 'B0', dicom_info.MagneticFieldStrength)
        nii = nib.Nifti1Image(chi_iLSQR, np.eye(4))
        nib.save(nii, '\\LBV\\chi_iLSQR_smvrad' + str(smv_rad) + '.nii')

        # % MEDI
        # ### normalize signal intensity by noise to get SNR %%%
        # ### Generate the Magnitude image %%%%
        # iMag = sqrt(sum(mag.^2,4));
        # % [iFreq_raw N_std] = Fit_ppm_complex(ph_corr);
        # matrix_size = single(imsize(1:3));
        # voxel_size = vox;
        # delta_TE = TE(2) - TE(1);
        # B0_dir = z_prjs';
        # CF = dicom_info.ImagingFrequency *1e6;
        # iFreq = [];
        # N_std = 1;
        # RDF = lfs_lbv*2.675e8*dicom_info.MagneticFieldStrength*delta_TE*1e-6;
        # Mask = mask_lbv;
        # save RDF.mat RDF iFreq iMag N_std Mask matrix_size...
        #         voxel_size delta_TE CF B0_dir;
        # QSM = MEDI_L1('lambda',1000);
        # nii = make_nii(QSM.*Mask,vox);
        # save_nii(nii,['LBV/MEDI1000_lbv_smvrad' num2str(smv_rad) '.nii']);
        # QSM = MEDI_L1('lambda',2000);
        # nii = make_nii(QSM.*Mask,vox);
        # save_nii(nii,['LBV/MEDI2000_lbv_smvrad' num2str(smv_rad) '.nii']);
        # QSM = MEDI_L1('lambda',1500);
        # nii = make_nii(QSM.*Mask,vox);
        # save_nii(nii,['LBV/MEDI1500_lbv_smvrad' num2str(smv_rad) '.nii']);
        # QSM = MEDI_L1('lambda',5000);
        # nii = make_nii(QSM.*Mask,vox);
        # save_nii(nii,['LBV/MEDI5000_lbv_smvrad' num2str(smv_rad) '.nii']);
        # QSM = MEDI_L1('lambda',500);
        # nii = make_nii(QSM.*Mask,vox);
        # save_nii(nii,['LBV/MEDI500_lbv_smvrad' num2str(smv_rad) '.nii']);
    #
        # % TVDI method
        # sus_lbv = tvdi(lfs_lbv,mask_lbv,vox,tv_reg,mag(:,:,:,end),z_prjs,inv_num);
        # nii = make_nii(sus_lbv.*mask_lbv,vox);
        # save_nii(nii,'LBV/sus_lbv.nii');
    # end

    # TFI TODO 有大量未知函数，跳过

    # save('all.mat','-v7.3');
    # cd(init_dir);
    # TODO 保存所有参数，具体未知
    savemat('all.mat', {'lfs_lbv': lfs_lbv, 'mask_lbv': mask_lbv, 'chi_iLSQR': chi_iLSQR})
    os.chdir(init_dir)


if __name__ == "__main__":
    path_mag = '../../DICOMs/swi_1mm_5TE_prisma4_r3_6'
    path_ph = '../../DICOMs/swi_1mm_5TE_prisma4_r3_7'
    qsm_r2s_prisma(path_mag, path_ph)
