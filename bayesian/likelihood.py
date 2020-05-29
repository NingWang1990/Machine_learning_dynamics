import sympy
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from error_checker import check_binary_oneD_array
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
import warnings

model_evaluation_implemented = ['test_MSE','test_MAE','cross_validate_MSE',
                                'cross_validate_MAE']

class GaussianLogLikelihood():

    def __init__(self,X,y,reg_normalization=False, gaussian_error_std=None, 
                 model_evaluation='test_MAE',
                 test_size=0.3, cv_nfolds=4,
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
        test_size........... float or int. It is only used when the model_evaluation 
                             is test_MSE or test_MAE.
                             If float, should be between 0.0 and 1.0 and 
                             represent the proportion of the dataset to include in the test split. 
                             If int, represents the absolute number of test samples.
                             default: 0.3
        cv_nfilds........... int, It is only used when the model_evaluation
                             is cross_validate_MSE or cross_validate_MAE.
                             It specifies the number of folds in the cross-validation.
                             It will be passed into cross_validate in sklearn.
        radom_state......... int or None, random state for shuffling before
                             applying the train-test split
        """                 
        X,y = self.check_convert_data(X,y)
        self.gaussian_error_std = gaussian_error_std
        #self.regressor = BayesianRidge()
        self.regressor = LinearRegression(fit_intercept=False)
        self.reg_normalization = reg_normalization 
        if reg_normalization == True:
            variables_mean_ = np.mean(X, axis=0, keepdims=True)
            variables_std_ = np.std(X, axis=0, keepdims=True)
            X = (X - variables_mean_) / variables_std_
            self.variables_mean_ = np.squeeze(variables_mean_)
            self.variables_std_ = np.squeeze(variables_std_)

        self.model_evaluation = model_evaluation
        if self.model_evaluation == 'test_MSE' or self.model_evaluation == 'test_MAE':
            if len(X) < 500:
                warnings.warn('There are not many data points, I suggest you to use cross-validation. '+\
                              'To do so, choose the model_evaluation be cross_validate_MSE or cross_validate_MAE.')
            self.X_train, self.X_test, self.y_train, self.y_test = \
                   train_test_split(X,y,random_state=random_state,test_size=0.3,)
            self.trainTestSplit = True
        elif self.model_evaluation == 'cross_validate_MSE' or self.model_evaluation == 'cross_validate_MAE':
            # the data will not be suffled in cross_validate. So shuffle the data here.
            self.cv_nfolds = cv_nfolds
            self.X, self.y = shuffle(X, y, random_state=random_state)
            self.trainTestSplit = False

    def __call__(self,binary_coef):
        
        """
        binary_coef...........list or one D numpy array containing only 0 or 1.
        
        return:
               mean loglikelihood of all samples, 
               proposal coefficients
        """
        coef = np.array(binary_coef)
        check_binary_oneD_array(coef)
        if not len(coef) == self.get_n_features():
            raise ValueError("length of the binary_coef doesn't match number of variables (columns) in X")
        mask = coef > 0.
        refit = True
        if hasattr(self, 'coef_treated') and np.all(self.coef_treated == coef):
            refit = False
        if refit is True:
            if self.model_evaluation is 'test_MSE' or self.model_evaluation is 'test_MAE': 
                error_v, reg_coef = self.regression(self.X_train,self.y_train,self.X_test,self.y_test,mask)
            else:
                error_v = []
                reg_coef = []
                #for train_index, test_index in self.kf_splits:
                for train_index,test_index in KFold(self.cv_nfolds).split(self.X, self.y):
                    X_train = self.X[train_index]
                    y_train = self.y[train_index]
                    X_test =  self.X[test_index]
                    y_test =  self.y[test_index]
                    err, rc = self.regression(X_train, y_train, X_test, y_test, mask)
                    error_v += list(err)
                    reg_coef += [rc,]
                reg_coef = np.array(reg_coef)
                reg_coef = np.mean(reg_coef, axis=0)
            error_v = np.array(error_v)
            if self.gaussian_error_std is None:
                std = np.sqrt(np.mean(error_v*error_v))
            else:
                std = self.gaussian_error_std
            self.log_likelihood_ =  np.mean(norm.logpdf(error_v,scale=std))
            if self.model_evaluation is 'test_MSE' or self.model_evaluation is 'cross_validate_MSE':
                self.model_metric_ = np.mean(error_v*error_v)
            elif self.model_evaluation is 'test_MAE'or self.model_evaluation is 'cross_validate_MAE':
                self.model_metric_ = np.mean(np.abs(error_v))
            # weights and bias 
            weights = np.zeros(len(coef), dtype=np.float64)
            bias = 0.
            weights[mask] = reg_coef[:]
            if self.reg_normalization is True:
                # put mean values into bias
                bias -= np.sum(weights * self.variables_mean_ / self.variables_std_)
                # rescale weights
                weights /= self.variables_std_
            self.regress_weights_ = weights
            self.regress_bias_ =  bias
            # to avoid redundant call
            self.coef_treated = coef.copy()

        return self.log_likelihood_

    def regression(self, X_train, y_train, X_test, y_test, mask):
        if np.sum(mask) == 0:
            return y_test, np.array([])
        self.regressor.fit(X_train[:,mask],y_train)
        y_pred = self.regressor.predict(X_test[:,mask])
        error_v = y_test - y_pred
        return error_v, self.regressor.coef_
    
    def get_n_features(self):
        if self.trainTestSplit is True:
            n_features = self.X_train.shape[1]
        else:
            n_features = self.X.shape[1]
        return n_features

    def get_model_metric(self,binary_coef):
        """
        return regression MSE for the linear model specified by binary_coef
        """
        coef = np.array(binary_coef)
        check_binary_oneD_array(coef)
        if not len(coef) == self.get_n_features():
            raise ValueError("length of the binary_coef doesn't match number of variables (columns) in X")
        log_likelihood = self.__call__(coef)
        return self.model_metric_

    def get_weights_bias(self,binary_coef):
        """
        return regression weights and bias for the linear model specified by binary_coef 
        """
        coef = np.array(binary_coef)
        check_binary_oneD_array(coef)
        if not len(coef) == self.get_n_features():
            raise ValueError("length of the binary_coef doesn't match number of variables (columns) in X")
        log_likelihood = self.__call__(coef)
        
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
        #stds = np.std(X, axis=0)
        #if np.any(stds == 0.):
        #    indices = np.where(stds == 0.)
        #    format_d = len(indices)*'%d'
        #    raise ValueError('feature(s) in column '+format_d +' has (have) zero variace. ' + \
        #            'They will influence the estimation of the bias term. Please remove it (them)' % tuple(indices) )
        return X, y

    
    @property
    def model_evaluation(self):
        return self._model_evaluation
    @model_evaluation.setter
    def model_evaluation(self,model_evaluation):
        if not model_evaluation in model_evaluation_implemented:
            raise ValueError('model_evaluation not implemented, choose one from ', model_evaluation_implemented)
        self._model_evaluation = model_evaluation

    #@property
    #def test_percentage(self):
    #    return self._test_percentage
    #@test_percentage.setter
    #def test_percentage(self, test_percentage):
    #    if not (test_percentage > 0. and test_percentage < 1. ):
    #        raise ValueError('test_percentage should be in (0.,1.)')
    #    self._test_percentage = test_percentage
