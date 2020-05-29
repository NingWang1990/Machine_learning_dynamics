
import numpy as np
from copy import deepcopy
from error_checker import check_binary_oneD_array 
from posterior import LogPosterior
import io

def acceptance_judgement(current_logposterior,proposal_logposterior):
    """
    current_logposterior...........log posterior for current state
    proposal_logposterior..........log posterior for the proposal state
    """
    acceptance_prob = min(1., np.exp(proposal_logposterior-current_logposterior))
    rand = np.random.random()
    acceptance = False
    if rand < acceptance_prob:
        acceptance = True
    return acceptance, acceptance_prob

class MCMC():
    def __init__(self, log_file=None ):
        """
        log_file..................None or string or a writable io.IOBase instance.
                                  None: no log file. don't write out anything
                                  string: name of the log file
        """
                
        if (log_file == None ) or (isinstance(log_file,str)) or ( isinstance(log_file, io.IOBase) and  log_file.writable() ):
            self.log_file = log_file
        else:
            raise TypeError('log_file must be None or a string or a writable io.IOBase instance')
        
    def one_step(self,current,posterior):
        """
        one step MCMC
        current............one D ndarray containing only 0 or 1.
        posterior.................object of LogPosterior
        """
        if not isinstance(posterior, LogPosterior):
            raise TypeError('posterior must be an object of ', LogPosterior)

        current = np.array(current)
        check_binary_oneD_array(current)

        rand = np.random.randint(0,len(current))
        proposal = current.copy()
        if proposal[rand] == 0.:
            proposal[rand] = 1.
        else:
            proposal[rand] = 0.
        current_logposterior = self.posterior(current)
        posterior_backup = deepcopy(self.posterior)
        proposal_logposterior = self.posterior(proposal)
        acceptance, prob = acceptance_judgement(current_logposterior,proposal_logposterior)
        if acceptance:
            return proposal, prob
        else:
            self.posterior = posterior_backup
            return current, prob
    
    def __call__(self,nsteps,current, posterior, log_header=False):
        """
        current............one D ndarray containing only 0 or 1.
        posterior.................object of LogPosterior
        """
        current = np.array(current)
        check_binary_oneD_array(current)
        if not isinstance(posterior, LogPosterior):
            raise TypeError('posterior must be an object of ', LogPosterior)
        self.posterior = posterior
        if not self.log_file == None:
            if isinstance(self.log_file, str):
                log_file = open(self.log_file,'a')
            else:
                log_file = self.log_file
        else:
            log_file = None
        # initialize prob
        prob = 1.
        if not log_file == None: 
            self.log(0, current, prob, log_file, log_header)
        for i in range(nsteps):
            current,prob = self.one_step(current,self.posterior)
            if not log_file == None:
                self.log(i+1, current, prob, log_file, log_header)
        
        if not self.log_file == None:
            if isinstance(self.log_file, str):
                log_file.close()

        return current 

    def log(self, step, current, prob, log_file,log_header=True):
        if not (isinstance(log_file, io.IOBase) and  log_file.writable()):
            raise TypeError('log_file must be writable io.IOBase instance')
               
        #log_posterior = self.posterior.log_posterior_
        #log_prior = self.posterior.prior.log_prior_
        #log_likelihood = self.posterior.likelihood.log_likelihood_
        log_posterior = self.posterior(current)
        log_prior = self.posterior.prior(current)
        log_likelihood = self.posterior.likelihood(current)
        weights,bias = self.posterior.likelihood.get_weights_bias(current)

        if step == 0:
            if log_header == True:
                log_format =  "% 10s  " + len(coef) * "  %- 16s  " + "  % 12s " + 3*"  % 15s  " + "\n"
                headers = ('step',)
                for i in range(len(coef)-1):
                    headers += ('weight_'+str(i+1),)
                headers += ('bias',)
                headers += ('accp_prob',)
                headers += ('logPrior','logLikelihood','logPosterior')
                log_file.write(log_format % headers)
                
        log_format = "% 10d  " + len(coef) * "  %- 16.8f  " + "  % 12.3f  " + 3*"  % 15.3f  " + "\n"
        data =  (step,) + tuple(weights) + (bias,) + (prob,) + (log_prior, log_likelihood, log_posterior)
        log_file.write(log_format % data)



