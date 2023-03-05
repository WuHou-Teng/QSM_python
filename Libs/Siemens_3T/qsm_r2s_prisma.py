import os

from tools.Exception_Def import ParamException


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


