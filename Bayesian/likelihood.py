import sympy
import numpy as np
from sklearn.linear_model import BayesianRidge
from scipy.stats import norm
from error_checker import check_binary_oneD_array

class GaussianLogLikelihood():

    def __init__(self,X,Y,reg_normalization=True):
        """
        X...................ndarray of shape (n_samples, n_variables)
        Y...................ndarray of shape (n_samples,)
        reg_normalization...Boolean. True: first normalize X and do regression.
        """
        self.X = np.array(X)
        self.Y = np.array(Y)
        if not self.X.ndim == 2:
            raise ValueError('X must be 2D array')
        if not self.X.shape[0] == self.Y.shape[0]:
            raise ValueError('length of X and Y must be identical')
        self.regressor = BayesianRidge()
        self.reg_normalization = reg_normalization 
        if reg_normalization == True:
            variables_mean_ = np.mean(X,axis=0, keepdims=True)
            variables_std_ = np.std(X, axis=0, keepdims=True)
            self.X = (self.X - variables_mean_) / variables_std_
            self.variables_mean_ = np.squeeze(variables_mean_)
            self.variables_std_ = np.squeeze(variables_std_)
        
    def __call__(self,binary_coef):
        
        """
        binary_coef...........list or one D numpy array containing only 0 or 1.
        
        return:
               mean loglikelihood of all samples, 
               proposal coefficients
        """
        coef = np.array(binary_coef)
        check_binary_oneD_array(coef)
        shape = self.X.shape
        if not len(coef) == shape[1]:
            raise ValueError("length of the binary_coef doesn't match number of variables (columns) in X")
        mask = coef > 0.
        n_samples = len(self.X)
        x = np.ones((n_samples,np.sum(mask)+1))  # add one column for constant shift.
        x[:,0:np.sum(mask)] = self.X[:,mask]
        y = self.Y
        refit = True
        if hasattr(self, 'coef_treated') and np.all(self.coef_treated == coef):
            refit = False
        if refit is True:
            self.regressor.fit(x,y)
            self.coef_treated = coef.copy()
        mean, std = self.regressor.predict(x, return_std=True)
        self.log_likelihood_ =  np.mean(norm.logpdf(y,loc=mean,scale=std))
        return self.log_likelihood_

    def get_regress_coef(self,binary_coef):
        coef = np.array(binary_coef)
        check_binary_oneD_array(coef)
        shape = self.X.shape
        if not len(coef) == shape[1]:
            raise ValueError("length of the binary_coef doesn't match number of variables (columns) in X")
        log_likelihood = self.__call__(coef)
        regress_coef = np.zeros( len(coef)+1, dtype=np.float64)
        mask = coef > 0.
        mask = np.append(mask,True)
        regress_coef[mask] = self.regressor.coef_
 
        if self.reg_normalization is True:
            # put mean values into bias
            regress_coef[-1] -= np.sum(regress_coef[:-1] * self.variables_mean_ / self.variables_std_)
            # rescale weights
            regress_coef[:-1] /= self.variables_std_
        
        self.regress_coef_ = regress_coef

        return self.regress_coef_

    @property 
    def X(self):
        return self._X
    @X.setter
    def X(self, X):
        stds = np.std(X, axis=0)
        if np.any(stds == 0.):
            indices = np.where(stds == 0.)
            format_d = len(indices)*'%d'
            raise ValueError('feature(s) in column '+format_d +' has (have) zero variace. ' + \
                    'They will influence the estimation of the bias term. Please remove it (them)' % tuple(indices) )
        self._X = X
