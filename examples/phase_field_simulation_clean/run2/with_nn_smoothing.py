#!/usr/bin/env python
# coding: utf-8
import numpy as np
import sys
sys.path.append('/u/wangnisn/devel/Machine_learning_dynamics/bayesian')
import scipy
from scipy.io import loadmat
from data_generator import DataGenerator
from likelihood import GaussianLogLikelihood
from prior import ComplexityLogPrior
from posterior import LogPosterior
from mcmc import MCMC
from sequential_mc import SequentialMC
from util import generate_binary_coef_random, generate_binary_coef_simple
from get_pareto import ParetoSet

random_seed = 1001
np.random.seed(random_seed)
data = np.load('../train_data_simulation_clean.npy')
dg = DataGenerator()
n_samples = 30000
np.random.shuffle(data)
data = data[:n_samples]

X, names, complexities = dg( data[:,:-1], descriptions=['u','u_x','u_xx'],term_order_max=[4,2,1] )
Y = data[:,-1]

prior = ComplexityLogPrior(method='num_terms', simplicity_preference=0.1,complexity_terms=complexities)
likelihood = GaussianLogLikelihood(X,Y,reg_normalization=False)
posterior = LogPosterior(prior, likelihood)

init_coef1 = generate_binary_coef_random(len(names),40)
init_coef2 = generate_binary_coef_simple(len(names),1)
init_coefs = []
init_coefs += list(init_coef1[:20])
init_coefs += list(init_coef2)

sequentialmc = SequentialMC(posterior=posterior,pareto_set=ParetoSet(),pareto_filename='pareto_set')
#init_coefs[0] = np.array([0,1,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
pareto_set = sequentialmc(samples=init_coefs,beta0_nsteps=100,beta0to1_nsteps=1000,beta1_nsteps=1000, mcmc_nsteps=10,feature_descriptions=names )
array = pareto_set.to_array()
pareto_set[0].data
pareto_set[1].data
pareto_set[2].data
pareto_set[3].data
pareto_set[4].data
pareto_set[5].data
pareto_set[6].data
pareto_set.save_csv('pareto_set_final.csv')
