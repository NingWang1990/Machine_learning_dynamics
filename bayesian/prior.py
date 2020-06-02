import numpy as np
from error_checker import check_binary_oneD_array

class ComplexityLogPrior():
    
    def __init__(self,simplicity_preference=0.,complexity_terms=1.):
        """
        simplicity_preference.....positive float. A larger value leads to a simpler expression.
                                                  the default value 0. means no preference and 
                                                  each expression has the same prior probability
        complexity_terms..........float or 1D array-like.  
                                  if float: every term has the same complexity
                                  if array-like, each entry specifies the complexity for the 
                                  corresponding term
        """

        self.simplicity_preference = simplicity_preference
        self.complexity_terms = complexity_terms
    
    def evaluate_complexity(self,binary_coef):
        coef = np.array(binary_coef)
        check_binary_oneD_array(coef)
        if isinstance(self.complexity_terms, np.ndarray):
            if not len(self.complexity_terms) == len(coef):
                raise ValueError('self.complexity_terms must have the same length with binary_coef')
        complexity = np.sum(coef * self.complexity_terms)
        return complexity

    def __call__(self,binary_coef):
        # unnormalized log prior probability
        coef = np.array(binary_coef)
        check_binary_oneD_array(coef)
        complexity = self.evaluate_complexity(coef)
        self.log_prior_ = -1. * self.simplicity_preference * complexity
        return self.log_prior_
                
        
    @property
    def complexity_terms(self):
        return self._complexity_terms
    @complexity_terms.setter
    def complexity_terms(self, complexity_terms):
        if isinstance(complexity_terms, float):
            self._complexity_terms = complexity_terms
        elif isinstance(complexity_terms, list) or isinstance(complexity_terms, np.ndarray):
            complexity_terms = np.array(complexity_terms)
            if not complexity_terms.ndim == 1:
                raise TypeError('complexity_terms can only be float or 1D array-like')
            self._complexity_terms = complexity_terms

            
