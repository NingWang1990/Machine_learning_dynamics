import numpy as np
from error_checker import check_binary_oneD_array

methods_implemented = ['num_terms']
class ComplexityLogPrior():
    
    def __init__(self,method='num_terms',simplicity_preference=0.):
        """
        method....................str, method to calculate complexity. 
                                  it should be the member of methods_implemented
        simplicity_preference.....positive float. A larger value leads to a simpler expression.
                                                  the default value 0. means no preference and 
                                                  each expression has the same prior probability 
        """

        if not method in methods_implemented :
            raise ValueError('method should be in ', methods_implemented )
        self.method = method
        self.simplicity_preference = simplicity_preference
    
    def evaluate_complexity(self,binary_coef):
        coef = np.array(binary_coef)
        check_binary_oneD_array(coef)
        if self.method is 'num_terms':
            complexity = float(np.sum(coef > 0.)) / len(coef)
        return complexity

    def __call__(self,binary_coef):
        # unnormalized log prior probability
        coef = np.array(binary_coef)
        check_binary_oneD_array(coef)
        complexity = self.evaluate_complexity(coef)
        self.log_prior_ = -1. * self.simplicity_preference * complexity
        return self.log_prior_
                
        

