import numpy as np
from scipy.ndimage import convolve

from Libs.Misc.makeodd import makeodd
from Libs.Misc.createellipsoid import createellipsoid


def shaver(ROIin, R):
    """
    Shaves a 3D binary mask by 'R'.

    Parameters:
    - ROIin: Input binary mask (3D numpy array)
    - R: Radius or radii for the ellipsoid (scalar or 3-component vector)

    Returns:
    - ROIout: Shaved binary mask (3D numpy array)
    """
    if np.any(R == 0):
        ROIout = ROIin
        print('Radius cannot be zero. Returning input array')
    else:
        inputIsFloat = True
        if ROIin.dtype in (np.float32, np.float64):
            inputIsFloat = True
            ROIin = ROIin.astype(bool)
        elif ROIin.dtype == np.bool:
            inputIsFloat = False

        if np.any(np.mod(ROIin.shape, 2) == 0):
            isCroppingToOddDimensions = True
            ROIin = makeodd(ROIin)
            gridDimensionVector = ROIin.shape
        else:
            isCroppingToOddDimensions = False
            gridDimensionVector = ROIin.shape

        sphere = createellipsoid(gridDimensionVector, R)
        ROIout = convolve(ROIin, sphere / np.sum(sphere))
        ROIout = np.abs(ROIout) >= 1 - 0.99 / np.sum(sphere)

        if inputIsFloat:
            ROIout = ROIout.astype(float)

        if isCroppingToOddDimensions:
            ROIout = makeodd(ROIout, 'isUndoing')

    return ROIout
