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
import pandas as pd

random_seed = 1001

np.random.seed(random_seed)

data = np.load('../train_data_experiment_2D.npy')


dg = DataGenerator()

n_half = 10000
np.random.shuffle(data)
data1 = data[:n_half]

mask =  data[:,-1] > 0.005
data2 = data[mask][:n_half]

data = np.concatenate([data1,data2],axis=0)

X, names, complexity = dg(data[:,:-1], descriptions=['u','u_x','u_y','u_xx','u_xy','u_yy'],term_order_max=[4,2,2,1,1,1] )
Y = data[:,-1]


prior = ComplexityLogPrior(method='num_terms', simplicity_preference=0.1, complexity_terms=1.)
likelihood = GaussianLogLikelihood(X,Y,reg_normalization=False,random_state=random_seed,model_evaluation='cross_validate_MAE',cv_nfolds=3)
posterior = LogPosterior(prior, likelihood)

init_coefs1 = generate_binary_coef_random(len(names),40)
init_coefs2 = generate_binary_coef_simple(len(names),1)
init_coefs = list(init_coefs1) + list(init_coefs2)

sequentialmc = SequentialMC(posterior=posterior,pareto_filename='pareto_set')

for i, name in enumerate(names):
    print (i, name)

pareto_set = sequentialmc(samples=init_coefs,beta0_nsteps=100,beta0to1_nsteps=1000,beta1_nsteps=1000, mcmc_nsteps=10,feature_descriptions=names,writeout_interval=100 )

#pareto_set.save_csv('pareto_set_final.csv')

