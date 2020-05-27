import numpy as np
import itertools
from numerical_derivative import ChebyshevLocalFit_1D, ChebyshevLocalFit_2D
from sympy.parsing.sympy_parser import parse_expr
import sympy
term_builders_implemented = ['multiplication']

class DataGenerator():
    def __init__(self,derivative_order=2, term_builders=['multiplication',],term_order_max=4):
        
        """
        derivative_order.... int, the highest order of derivatives to be included in the expression
        term_builders........list of str. All elements should be in term_buliders_implemented.
                             used to build terms.
        term_order_max.......int, maximum term order 
        width................int, the width to fit the 
        """

        for tb in term_builders:
            if not tb in term_builders_implemented:
                raise ValueError( tb, " not implemented. Choose term_builders from ",term_builders_implemented )
        
        self.term_builders = term_builders
        self.term_order_max = term_order_max

    def __call__(self, data,descriptions):
        
        """
        data................. ndarray of shape (n_samples, n_variables)
        descriptions......... list of string or sympy expressions, descriptions of each column (variable)
        """

        """
        def check_data(z, spatial_grid, time_grid):
            shape_z = z.shape
            shape_spatial_grid = spatial_grid.shape
            shape_time_grid = time_grid.shape
            if not shape_z[:-1] == shape_spatial_grid:
                raise ValueError("shapes of z and spatial_grid don't match")
            if not shape_z[-1] = shape_time_grid[0]:
                raise ValueError("shapes of z and time_grid don't match")
        """
        data = np.array(data)
        if not data.ndim  == 2:
            raise ValueError('data should be 2D array')
        if not data.shape[1] == len(descriptions):
            raise ValueError('length of descriptions should be identical to number of columns in data')
        sympy_exprs = []
        for desp in descriptions:
            if isinstance(desp, str):
                try:
                    expr = parse_expr(desp)
                    sympy_exprs += [expr,]
                except:
                    raise ValueError('%s cannot be converted to sympy expressions', desp)
            elif isinstance(desp, sympy.Expr):
                sympy_exprs += [desp,]
            else:
                raise TypeError('elements of descriptions should be string or sympy expression')

        n_samples = data.shape[0]
        n_variables = data.shape[1]
        terms_all = np.ones((n_samples,1))
        names_all = [sympy.Integer(1),]
        for builder in self.term_builders:
            if builder == 'multiplication':
                var_list = np.arange(n_variables)
                combinations = []
                for i in range(1,len(var_list)+1):
                    combinations += list(itertools.combinations_with_replacement(var_list, i))
                terms = np.ones((n_samples,len(combinations)))
                names = len(combinations)*[parse_expr('1'),]
                for i,comb in enumerate(combinations):
                    for j in comb:
                        terms[:,i] *= data[:,j]
                        names[i] *= sympy_exprs[j]
                terms_all = np.concatenate([terms_all, terms],axis=1)
                names_all += names

        #for i in range(len(names)):
        #    names[i] = str(names[i])
        return terms_all, names_all




        

    
