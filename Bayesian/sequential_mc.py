import numpy as np
from posterior import LogPosterior
from mcmc import MCMC 
import sympy
import io
import csv

class SequentialMC():
    def __init__(self, posterior, mcmc=None,log_file=None):
        """
        posterior...........object of the class LogPosterior
        mcmc................None or object of the class MCMC
        log_file............None or string or a writable io.IOBase instance.
                            None: no log file. don't write out anything
                            string: name of the log file
        """
        if (log_file == None ) or (isinstance(log_file,str)) or ( isinstance(log_file, io.IOBase) and  log_file.writable() ):
            self.log_file = log_file
        else:
            raise TypeError('log_file must be None or a string or a writable io.IOBase instance')

        self.posterior = posterior
        self.mcmc = mcmc

    def __call__(self, samples, descriptions, beta0_nsteps=100, beta0to1_nsteps=100,beta1_nsteps=100,mcmc_nsteps=100):
        """
        samples.....................ndarray of shape (n_samples, n_features)
                                    initial samples to perform sequential MC  
        descriptions............... list of sympy expressions for each feature (column) in samples
        beta0_nsteps................int, # of steps with beta = 0
        beta0to1_nsteps.............int, # of steps with beta 0->1
        beta1_nsteps................int, # of steps with beta = 1
        mcmc_nsteps.................int, # of mcmc steps
        descriptions......... list of sympy expressions, descriptions of each column (variable)
        """
        for desp in descriptions:
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
            self.log(0, samples, descriptions, out_csv)
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
            for j,sample in enumerate(re_samples):
                samples[j] = self.mcmc(nsteps=mcmc_nsteps, current=sample,posterior=self.posterior,log_header=False)
            
            if not out_csv == None:
                self.log(i+1, samples, descriptions, out_csv)
        if not self.log_file == None:
            if isinstance(self.log_file, str):
                log_file.close()
        
        return samples

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

    def log(self, step, samples, descriptions, out_csv, log_header=True):
        """
        step................. int 
        samples.,,,,,,,,..... 2D array-like of shape (n_samples, n_features)
        descriptions......... list of sympy expressions, descriptions of each feature (column)
        out_csv.............. csv.writer object
        """
        if step == 0:
            if log_header is True:
                headers = ['step']
                for i in range(len(samples)):
                    headers += ['sample '+str(i),]
                out_csv.writerow(headers)

        row = [step,]
        for sample in samples:
            coef = self.posterior.likelihood.get_regress_coef(sample)
            expr = sympy.Integer(0)
            if not len(coef) == len(descriptions) + 1:
                raise ValueError("lengths of coef and descriptions don't match")
            for i, desp in enumerate(descriptions):
                expr += coef[i] * desp
            expr += coef[-1]
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

