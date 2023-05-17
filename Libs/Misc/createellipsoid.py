import numpy as np


def createellipsoid(gridDimensions, radii, offset=None):
    """
    Creates an ellipsoidal region (cells == 1) within an otherwise 0 array.

    Parameters:
    - gridDimensions: Size of the array (3-element list or tuple)
    - radii: Radii of the ellipsoid (3-element list or tuple)
    - offset: Offset of the ellipsoid center (3-element list or tuple)

    Returns:
    - ellipsoid: Array with the ellipsoid region

    """
    # Check arguments
    if offset is None:
        offset = [0, 0, 0]
    if len(offset) == 1:
        offset = offset * [1, 1, 1]

    # Create ellipsoid
    x, y, z = np.ogrid[-radii[0]:radii[0] + 1, -radii[1]:radii[1] + 1, -radii[2]:radii[2] + 1]  # coordinates
    ellipsoid = (x ** 2 / radii[0] ** 2 + y ** 2 / radii[1] ** 2 + z ** 2 / radii[2] ** 2) <= 1  # equation of ellipsoid

    # Place it in a larger array
    temp = np.zeros(gridDimensions)
    origin = (np.array(gridDimensions) + [1, 1, 1]) // 2
    offset = origin + offset
    temp[offset[0] - radii[0]:offset[0] + radii[0] + 1, offset[1] - radii[1]:offset[1] + radii[1] + 1,
    offset[2] - radii[2]:offset[2] + radii[2] + 1] = ellipsoid

    ellipsoid = temp
    return ellipsoid
