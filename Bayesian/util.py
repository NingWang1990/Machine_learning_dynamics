import numpy as np 
import itertools


def generate_binary_coef(max_non_zeros=1, length=10):
    """
    max_non_zeros........non-negative int, maximum non-zero values
    length...............int, length of the binary coefficients

    return:  list of binary coefficient vector with dtype np.int32
    """
    if max_non_zeros > length:
        raise ValueError('max_non_zeros must be not greater than length')

    coef_all = [np.zeros(length, dtype=np.int32)]

    for non_zeros in range(1, max_non_zeros+1):
        combinations = list(itertools.combinations(np.arange(length), non_zeros))
        for comb in combinations:
            coef = np.zeros(length, dtype=np.int32)
            for i in comb:
                coef[i] = 1
            coef_all += [coef,]
    
    return coef_all
