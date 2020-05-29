import numpy as np
from prior import ComplexityLogPrior
from likelihood import GaussianLogLikelihood
from error_checker import check_binary_oneD_array   

class LogPosterior():
    def __init__(self,prior,likelihood,beta=1.):
        """
        prior...............an object of class ComplexityLogPrior
        likelihood..........an object of class GaussianLogLikelihood
        beta....................tempering parameter in the range [0, 1]
        """

        if not isinstance(prior, ComplexityLogPrior):
            raise TypeError('prior must be an object of ', ComplexityLogPrior)
        if not isinstance(likelihood, GaussianLogLikelihood):
            raise TypeError('likelihood must be an object of ',GaussianLogLikelihood)

        self.prior = prior
        self.likelihood = likelihood
        self.beta = beta

    def __call__(self, binary_coef):
        """
        binary_coef...........list or one D numpy array containing only 0 or 1. 
                """
        coef = np.array(binary_coef)
        check_binary_oneD_array(coef)
        logprior = self.prior(coef)
        if self.beta == 0.:
            self.log_posterior_ = logprior
        else:
            loglikelihood = self.likelihood(coef)
            self.log_posterior_ = self.beta*loglikelihood + logprior 
        return self.log_posterior_
