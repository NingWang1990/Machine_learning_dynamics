import numpy as np

def check_binary_oneD_array(array):
    if not isinstance(array,np.ndarray):
        raise TypeError('array not a numpy.ndarray',array)
    if not array.ndim == 1:
        raise TypeError('array not one-dimensional', array)
    if not np.sum( (array == 0) + (array == 1) ) == len(array):
        raise ValueError('array contains values other than 0 and 1.',array)

