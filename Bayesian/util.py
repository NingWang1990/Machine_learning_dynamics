import numpy as np 
import itertools
from sympy import count_ops
import sympy
from sympy.parsing.sympy_parser import parse_expr

def expression_complexity(expr):
    """
    expr............string or sympy expression
    """
    if isinstance(expr, str):
        expr = parse_expr(expr)
    elif not isinstance(expr, sympy.Expr):
        raise TypeError('expr can only be sympy exprsssion or string')
    n_operations = count_ops(expr)
    return 1 + n_operations

def generate_binary_coef_random(length, n_samples=10):
    """
    Generate a list of binary coefficient vectors randomly
    length.............. int, length of the binary coefficients
    n_samples........... int, number of vectors

    return: 2D ndarray with each row a  binary coefficient vector.
            shape: (n_samples, length)
            dtype: np.int32
    """
    return np.random.randint(low=0,high=2,size=(n_samples, length),dtype=np.int32) 



def generate_binary_coef_simple(length, max_non_zeros=1):
    """
    Generate all possible binary coefficient vectors with an user-specified 
    maximum number of non-zero entries.
    
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
