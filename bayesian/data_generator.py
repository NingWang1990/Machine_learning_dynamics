import numpy as np
import itertools
from numerical_derivative import ChebyshevLocalFit_1D, ChebyshevLocalFit_2D
from sympy.parsing.sympy_parser import parse_expr
import sympy
term_builders_implemented = ['multiplication']
from util import expression_complexity

class DataGenerator():
    def __init__(self, term_builders=['multiplication',]):
        
        """
        term_builders........list of str. All elements should be in term_buliders_implemented.
                             Iused to build terms.
                """

        for tb in term_builders:
            if not tb in term_builders_implemented:
                raise ValueError( tb, " not implemented. Choose term_builders from ",term_builders_implemented )
        
        self.term_builders = term_builders

    def __call__(self, data,descriptions,term_order_max=4,variables_order_max=None):
        
        """
        data................. ndarray of shape (n_samples, n_variables)
        descriptions......... list of string or sympy expressions, descriptions of each column (variable)
        term_order_max....... int, specify the same maximum order for each term.
        variables_order_max.. None or 1D array-like.
                              if list, specify maximum order for each variable.
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
        if not isinstance(term_order_max, int):
            raise TypeError('term_order_max must be int')
        if variables_order_max is not None:
            variables_order_max = np.array(variables_order_max, np.int64)
            if (not variables_order_max.ndim == 1):
                raise TypeError('variables_order_max must be 1D array-like')
            elif (not len(variables_order_max) == len(sympy_exprs)):
                raise TypeError('length of variables_order_max inconsistent with descriptions')

        n_samples = data.shape[0]
        n_variables = data.shape[1]
        terms_all = np.ones((n_samples,1))
        names_all = [sympy.Integer(1),]
        for builder in self.term_builders:
            if builder == 'multiplication':
                var_list = np.arange(n_variables)
                combinations = []
                # generate all possible combinations
                for i in range(1,term_order_max+1):
                    combinations += list(itertools.combinations_with_replacement(var_list, i))
                if variables_order_max is not None:
                    # to remove combinations that exceeds maximum order for variables
                    indices_del = []
                    for i,order in enumerate(variables_order_max):
                        for j, comb in enumerate(combinations):
                            tt = np.array(comb, np.int64)
                            if np.sum(tt==i) > order:
                                indices_del.append(j)
                    indices_del = np.unique(indices_del)
                    for i in sorted(indices_del, reverse=True):
                        del combinations[i]

                terms = np.ones((n_samples,len(combinations)))
                names = len(combinations)*[parse_expr('1'),]
                for i,comb in enumerate(combinations):
                    for j in comb:
                        terms[:,i] *= data[:,j]
                        try:
                            names[i] *= sympy_exprs[j]
                        except:
                            print ('failed to multiply names by:', sympy_exprs[j])
                            print ('most likely it is because the improper string description:', descriptions[j])
                            print ('try to use a different description and rerun this function')
                            raise
                terms_all = np.concatenate([terms_all, terms],axis=1)
                names_all += names

        complexities = []
        for name in names_all:
            complexities += [expression_complexity(name),]
        #for i in range(len(names)):
        #    names[i] = str(names[i])
        return terms_all, names_all, complexities




        

    
