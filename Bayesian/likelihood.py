import sympy
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from error_checker import check_binary_oneD_array
from sklearn.model_selection import train_test_split

class GaussianLogLikelihood():

    def __init__(self,X,y,reg_normalization=False, gaussian_error_std=None, test_size=0.3,
                 random_state=None):
        """
        X................... 2D array-like of shape (n_samples, n_variables)
        y................... 1D array-like of shape (n_samples,)
        reg_normalization... Boolean. True: first normalize X and do regression.
        guassian_error_std.. float or None, user-specified standard deviation of the 
                             Gaussian error distribution.
                             None: use the std estimated by the GaussianRidge to
                                   estimate Gaussian likelihood
                             float: the std from GaussianRidge will be ignored,
                                    and this value is used to estimate Gaussian 
                                    likelihood.
        test_size........... float or int, If float, should be between 0.0 and 1.0 and 
                             represent the proportion of the dataset to include in the test split. 
                             If int, represents the absolute number of test samples.
                             default: 0.3
        radom_state......... int or None, random state for shuffling before
                             applying the train-test split
        """                 
        X,y = self.check_convert_data(X,y)
        self.gaussian_error_std = gaussian_error_std
        #self.regressor = BayesianRidge()
        self.regressor = LinearRegression()
        self.reg_normalization = reg_normalization 
        if reg_normalization == True:
            variables_mean_ = np.mean(X, axis=0, keepdims=True)
            variables_std_ = np.std(X, axis=0, keepdims=True)
            X = (X - variables_mean_) / variables_std_
            self.variables_mean_ = np.squeeze(variables_mean_)
            self.variables_std_ = np.squeeze(variables_std_)
        self.X_train, self.X_test, self.y_train, self.y_test = \
               train_test_split(X,y,random_state=random_state,test_size=test_size,)
       
    def __call__(self,binary_coef):
        
        """
        binary_coef...........list or one D numpy array containing only 0 or 1.
        
        return:
               mean loglikelihood of all samples, 
               proposal coefficients
        """
        coef = np.array(binary_coef)
        check_binary_oneD_array(coef)
        shape = self.X_train.shape
        if not len(coef) == shape[1]:
            raise ValueError("length of the binary_coef doesn't match number of variables (columns) in X")
        mask = coef > 0.
        n_samples = len(self.X_train)
        x = np.ones((n_samples,np.sum(mask)))
        x[:,0:np.sum(mask)] = self.X_train[:,mask]
        y = self.y_train
        refit = True
        if hasattr(self, 'coef_treated') and np.all(self.coef_treated == coef):
            refit = False
        if refit is True:
            self.regressor.fit(x,y)
            self.coef_treated = coef.copy()
        # calculate MSE and likelihood in the test dataset
        x = np.ones((len(self.X_test),np.sum(mask)))
        x[:,0:np.sum(mask)] = self.X_test[:,mask]
        y = self.y_test
        mean = self.regressor.predict(x)
        if self.gaussian_error_std is None:
            std = np.std(mean-y)
        else:
            std = self.gaussian_error_std
        self.regression_MSE_ = np.mean(np.sum((mean-y)**2))
        #mean, std = self.regressor.predict(x, return_std=True)
        self.log_likelihood_ =  np.mean(norm.logpdf(y,loc=mean,scale=std))
        return self.log_likelihood_

    def get_regression_MSE(self,binary_coef):
        """
        return regression MSE for the linear model specified by binary_coef
        """
        coef = np.array(binary_coef)
        check_binary_oneD_array(coef)
        shape = self.X_train.shape
        if not len(coef) == shape[1]:
            raise ValueError("length of the binary_coef doesn't match number of variables (columns) in X")
        log_likelihood = self.__call__(coef)
        return self.regression_MSE_

    def get_weights_bias(self,binary_coef):
        """
        return regression weights and bias for the linear model specified by binary_coef 
        """
        coef = np.array(binary_coef)
        check_binary_oneD_array(coef)
        shape = self.X_train.shape
        if not len(coef) == shape[1]:
            raise ValueError("length of the binary_coef doesn't match number of variables (columns) in X")
        log_likelihood = self.__call__(coef)
        weights = np.zeros( len(coef), dtype=np.float64)
        bias = 0.
        mask = coef > 0.
        mask = np.append(mask,True)
        weights[mask] = self.regressor.coef_
        bias = self.regressor.intercept_
        if self.reg_normalization is True:
            # put mean values into bias
            bias -= np.sum(weights * self.variables_mean_ / self.variables_std_)
            # rescale weights
            weights /= self.variables_std_
        
        self.regress_weights_ = weights
        self.regress_bias_ =  bias

        return self.regress_weights_, self.regress_bias_

    
    def check_convert_data(self, X, y):
        """
        check data and convert them to ndarray
        """
        X = np.array(X)
        y = np.array(y)
        if not X.ndim == 2:
            raise ValueError('X must be 2D array-like')
        if not X.shape[0] == y.shape[0]:
            raise ValueError('length of X and y must be identical')
        stds = np.std(X, axis=0)
        if np.any(stds == 0.):
            indices = np.where(stds == 0.)
            format_d = len(indices)*'%d'
            raise ValueError('feature(s) in column '+format_d +' has (have) zero variace. ' + \
                    'They will influence the estimation of the bias term. Please remove it (them)' % tuple(indices) )
        return X, y


    @property
    def test_percentage(self):
        return self._test_percentage
    @test_percentage.setter
    def test_percentage(self, test_percentage):
        if not (test_percentage > 0. and test_percentage < 1. ):
            raise ValueError('test_percentage should be in (0.,1.)')
        self._test_percentage = test_percentage
