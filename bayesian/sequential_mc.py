import numpy as np
from posterior import LogPosterior
from mcmc import MCMC 
import sympy
import io
import csv
from get_pareto import ParetoPoint, ParetoSet
from multiprocessing import Pool
from functools import partial
class SequentialMC():
    def __init__(self, posterior, mcmc=None,log_file=None,pareto_filename=None, pareto_set=ParetoSet(),parallel=False):
        """
        posterior........... object of the class LogPosterior
        mcmc................ None or object of the class MCMC
        log_file............ None or string or a writable io.IOBase instance.
                             None: no log file. don't write out anything
                             string: name of the log file
        pareto_filename..... None or string.
                             None: don't write out Pareto set
                             string, write out Pareto set. filename: this string@str(step).csv
        pareto_set.......... object of the class PaeretoSet. 
                             the initial pareto set. The new points generated in __call__
                             will be added into it.
        """
        if (log_file == None ) or (isinstance(log_file,str)) or ( isinstance(log_file, io.IOBase) and  log_file.writable() ):
            self.log_file = log_file
        else:
            raise TypeError('log_file must be None or a string or a writable io.IOBase instance')
        
        if (pareto_filename==None) or isinstance(pareto_filename,str):
            self.pareto_filename = pareto_filename
        else:
            raise TypeError('log_file must be None or a string or a writable io.IOBase instance')

            
        if not isinstance(pareto_set, ParetoSet):
            raise TypeError('init_pareto_set must be an object of ', ParetoSet)

        self.posterior = posterior
        self.mcmc = mcmc
        self.pareto_set = pareto_set
        self.parallel = parallel

    def __call__(self, samples, feature_descriptions, beta0_nsteps=100, beta0to1_nsteps=100,beta1_nsteps=100,mcmc_nsteps=100,writeout_interval=100):
        """
        samples.....................ndarray of shape (n_samples, n_features)
                                    initial samples to perform sequential MC  
        feature_descriptions....... list of sympy expressions for each feature (column) in samples
        beta0_nsteps................int, # of steps with beta = 0
        beta0to1_nsteps.............int, # of steps with beta 0->1
        beta1_nsteps................int, # of steps with beta = 1
        mcmc_nsteps.................int, # of mcmc steps
        """
        for desp in feature_descriptions:
            if not isinstance(desp, sympy.Expr):
                raise ValueError('entry in descriptions must be sympy expression')
        if not self.log_file == None:
            if isinstance(self.log_file, str):
                log_file = open(self.log_file,'a', newline='')
            else:
                log_file = self.log_file
            out_csv = csv.writer(log_file)
        else:
            out_csv = None
        samples = np.array(samples)
        size = len(samples)
        betas = self.generate_betas(beta0_nsteps, beta0to1_nsteps,beta1_nsteps)
        if not out_csv == None:
            self.log(0, samples, feature_descriptions, out_csv)
    
        for sample in samples:
            self.add_to_ParetoSet(sample, feature_descriptions)
    
        
        for i,beta in enumerate(betas):
            print ('step: %d, beta: %6.3f' % (i, beta))
            # set beta
            self.posterior.beta = beta
            #reweighting 
            uniques, frequencies = self.unique(samples)
            log_posteriors = self.log_posterior(beta, uniques)
            p = self.reweight(log_posteriors, frequencies)
            #resampling
            re_samples = self.resample(uniques,size,p )
            # mcmc
            if self.parallel is True:
                mcmc_func = partial(self.mcmc, nsteps=mcmc_nsteps, posterior=self.posterior, log_header=False)
                pool = Pool()
                samples = pool.map(mcmc_func, list(re_samples))
            else:
                # serial 
                for j,sample in enumerate(re_samples):
                    samples[j] = self.mcmc(nsteps=mcmc_nsteps, current=sample,posterior=self.posterior,log_header=False)
            
            for sample in samples:
                self.add_to_ParetoSet(sample, feature_descriptions)
            
            if i%writeout_interval == 0:
                if not out_csv == None:
                    self.log(i+1, samples, feature_descriptions, out_csv)
                if not self.pareto_filename is None:
                    self.pareto_set.save_csv(self.pareto_filename+'@'+str(i)+'.csv')      
            

        if not self.log_file == None:
            if isinstance(self.log_file, str):
                log_file.close()
        
        return self.pareto_set

    def add_to_ParetoSet(self, sample, feature_descriptions):
        """
        add sample to ParetoSet
        """
        complexity = self.posterior.prior.evaluate_complexity(sample)
        mse = self.posterior.likelihood.get_model_metric(sample)
        weights,bias = self.posterior.likelihood.get_weights_bias(sample)
        expr = self.construct_linear_expression(weights, bias,feature_descriptions)
        pareto_point = ParetoPoint(x=complexity, y=mse, data=expr)
        self.pareto_set.add(pareto_point)
        
    def log_posterior(self,beta,samples):
        """
        samples.................ndarray of shape (n_samples, n_features)
        beta....................float, tempering parameter

        return: 1D ndarray for log_posterior of all samples at the given
                termpering parameter beta.
        """
        log_posteriors = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            log_prior = self.posterior.prior(sample)
            log_likelihood = self.posterior.likelihood(sample)
            log_posteriors[i] = beta * log_likelihood + log_prior
        return log_posteriors
            
    def generate_betas(self, beta0_nsteps, beta0to1_nsteps,beta1_nsteps):
        n0 = beta0_nsteps
        n1 = beta0to1_nsteps
        n2 = beta1_nsteps
        betas = np.zeros(n0+n1+n2)
        betas[n0:(n0+n1)] = np.linspace(0.,1., n1)
        betas[(n0+n1):(n0+n1+n2)] = 1.
        return betas

    def resample(self, samples, size, p):
        """
        samples...........2d array-like. shape: (n_samples, n_features)
        size..............int, # of random samples to draw
        p.................1D array-like, the probabilities associated with each
                          sample. its length must be identical with n_samples
        """
        samples = np.array(samples)
        if not samples.ndim == 2:
            raise TypeError('samples must be 2d array-like')
        indices = np.random.choice(np.arange(len(samples)), size=size, p=p)
        return samples[indices,:] 

    def reweight(self,log_posteriors, frequencies):
        """
        return probabilities after reweighting
        """
        w = log_posteriors + np.log(frequencies)
        w_sum = np.log(np.sum(np.exp(w - np.max(w)))) + np.max(w)  # avoid underflow
        p = np.exp(w - w_sum)
        return p

    def unique(self, X):
        """
        X..........ndarry of shape (n_samples, n_features)
        """
        return np.unique(X, axis=0, return_counts=True)

    def construct_linear_expression(self, weights, bias, feature_descriptions):
        """
        evaluate the expression: weights * feature_descriptions.T + bias with sympy

        weights................... 1D array-like, weights of each feature
        bias...................... float, bias
        feature_descriptions...... list of sympy expressions, descriptions of each feature.
        
        Return:
        sympy expression
        
        """
        #for desp in feature_descriptions:
        #    if not isinstance(desp, sympy.Expr):
        #        raise ValueError('entry in descriptions must be sympy expression')            
        if not len(weights) == len(feature_descriptions):
            raise ValueError('weights must have the same length with feature_descriptions')
        expr = sympy.Integer(0)
        for i, desp in enumerate(feature_descriptions):
            expr += weights[i] * desp
        expr += bias
        return expr
        
    def log(self, step, samples, feature_descriptions, out_csv, log_header=True):
        """
        step.......................... int 
        samples.,,,,.........,,,,..... 2D array-like of shape (n_samples, n_features)
        feature__descriptions......... list of sympy expressions, descriptions of each feature (column)
        out_csv....................... csv.writer object
        """
        if step == 0:
            if log_header is True:
                headers = ['step']
                for i in range(len(samples)):
                    headers += ['sample '+str(i),]
                out_csv.writerow(headers)

        row = [step,]
        for sample in samples:
            weights, bias = self.posterior.likelihood.get_weights_bias(sample)
            expr = self.construct_linear_expression(weights,bias,feature_descriptions)
            row += [ str(expr),]
        out_csv.writerow(row)

    # data encapsulation
    @property
    def mcmc(self):
        return self._mcmc
    @mcmc.setter
    def mcmc(self,mcmc):
        if not mcmc is None:
            if not isinstance(mcmc, MCMC):
                raise TypeError('mcmc must be an object of ', MCMC)
            self._mcmc = mcmc
        else:
            self._mcmc = MCMC()

    @property
    def posterior(self):
        return self._posterior
    @posterior.setter
    def posterior(self, posterior):
        if not isinstance(posterior, LogPosterior):
            raise TypeError('posterior must be an object of ', LogPosterior)
        self._posterior = posterior
