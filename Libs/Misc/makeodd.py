import numpy as np

def makeodd(dataArray, option='isShaving', finalGridDimensionVector=None):
    # option: 'isShaving', 'isUndoing'
    # finalGridDimensionVector: final array size

    if option == 'isUndoing':
        isPadding = True
    else:
        option = 'isShaving'
        isPadding = False

    gridDimensionVector = dataArray.shape

    isOdd = np.mod(gridDimensionVector, 2)
    isEven = np.logical_not(isOdd).astype(int)

    if finalGridDimensionVector is not None:
        numSlices = np.abs(gridDimensionVector - finalGridDimensionVector)
    else:
        if option == 'isShaving':
            numSlices = isEven
        elif option == 'isUndoing':
            numSlices = isOdd

    if isPadding:
        dataArray = np.pad(dataArray, [(0, numSlices[0]), (0, numSlices[1]), (0, numSlices[2])], mode='constant')
    else:
        dataArray = dataArray[:dataArray.shape[0] - numSlices[0], :dataArray.shape[1] - numSlices[1], :dataArray.shape[2] - numSlices[2]]

    return dataArray
