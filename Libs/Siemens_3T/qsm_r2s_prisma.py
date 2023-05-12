import os
import pydicom
from tools.Exception_Def import ParamException
from tools.StructureCLass import Options


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

    # if ~ isfield(options,'t_svd')
    #     options.t_svd = 0.1;
    # end
    if not options.is_field('t_svd'):
        options.t_svd = 0.1

    # if ~ isfield(options,'smv_rad')
    #     options.smv_rad = 3;
    # end
    if not options.is_field('smv_rad'):
        options.smv_rad = 3

    # if ~ isfield(options,'tik_reg')
    #     options.tik_reg = 1e-3;
    # end
    if not options.is_field('tik_reg'):
        options.tik_reg = 1e-3

    # if ~ isfield(options,'cgs_num')
    #     options.cgs_num = 500;
    # end
    if not options.is_field('cgs_num'):
        options.cgs_num = 500

    # if ~ isfield(options,'lbv_tol')
    #     options.lbv_tol = 0.01;
    # end
    if not options.is_field('lbv_tol'):
        options.lbv_tol = 0.01

    # if ~ isfield(options,'lbv_peel')
    #     options.lbv_peel = 2;
    # end
    if not options.is_field('lbv_peel'):
        options.lbv_peel = 2

    # if ~ isfield(options,'tv_reg')
    #     options.tv_reg = 5e-4;
    # end
    if not options.is_field('tv_reg'):
        options.tv_reg = 5e-4

    # if ~ isfield(options,'inv_num')
    #     options.inv_num = 500;
    # end
    if not options.is_field('inv_num'):
        options.inv_num = 500

    # if ~ isfield(options,'interp')
    #     options.interp = 0;
    # end
    if not options.is_field('interp'):
        options.interp = 0

    # % read in DICOMs of both magnitude and raw unfiltered phase images
    # % read in magnitude DICOMs
    # path_mag = cd(cd(path_mag)); TODO python不需要在文件夹之间来回跳，只需要直到正确的路径即可。
    # mag_list = dir(path_mag);
    mag_list = os.listdir(path_mag)  # 获取目录下所有文件的名称列表。
    # mag_list = mag_list(~strncmpi('.', {mag_list.name}, 1));
    mag_list = [filename for filename in mag_list if not filename.startswith('.')]  # 去除隐藏文件

    # % get the sequence parameters
    # dicom_info = dicominfo([path_mag,filesep,mag_list(end).name]);
    dicom_info = pydicom.dcmread(os.path.join(path_mag, mag_list[-1]))  # 读取dicom文件
    # EchoTrainLength = dicom_info.EchoNumbers;
    # for i = 1:EchoTrainLength % read in TEs
    #     dicom_info = dicominfo([path_mag,filesep,mag_list(1+(i-1)*(length(mag_list))./EchoTrainLength).name]);
    #     TE(dicom_info.EchoNumbers) = dicom_info.EchoTime*1e-3;
    # end
    # vox = [dicom_info.PixelSpacing(1), dicom_info.PixelSpacing(2), dicom_info.SliceThickness];

