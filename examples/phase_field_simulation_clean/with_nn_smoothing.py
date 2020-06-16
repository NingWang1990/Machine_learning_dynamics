#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import sys


# In[3]:


sys.path.append('/u/wangnisn/devel/Machine_learning_dynamics/bayesian')


# In[4]:


import scipy
from scipy.io import loadmat
from data_generator import DataGenerator


# In[361]:


from likelihood import GaussianLogLikelihood
from prior import ComplexityLogPrior
from posterior import LogPosterior
from mcmc import MCMC
from sequential_mc import SequentialMC
from util import generate_binary_coef_random, generate_binary_coef_simple
import pandas as pd


# In[6]:


import matplotlib.pyplot as plt


# In[43]:


random_seed = 1001
np.random.seed(random_seed)


# In[493]:


data = np.load('train_data_simulation_clean.npy')


# In[494]:


dg = DataGenerator()


# In[495]:


data[:,:-1].shape


# In[496]:


data.shape


# In[497]:


n_samples = 30000


# In[498]:


np.random.shuffle(data)
data = data[:n_samples]


# In[499]:


data.shape


# In[500]:


X, names, complexities = dg( data[:,:-1], descriptions=['u','u_x','u_xx'],term_order_max=[4,2,1] )


# In[501]:


X.shape


# In[502]:


Y = data[:,-1]
#Y = -0.9*X[:,0] + 10.*X[:,2] +1.9*X[:,3] -1.*X[:,9]


# In[527]:


prior = ComplexityLogPrior(method='num_terms', simplicity_preference=1.,complexity_terms=complexities)


# In[528]:


likelihood = GaussianLogLikelihood(X,Y,reg_normalization=False)


# In[529]:


posterior = LogPosterior(prior, likelihood)


# In[530]:


import itertools


# In[531]:


from get_pareto import ParetoSet


# In[532]:


init_coef1 = generate_binary_coef_random(len(names),40)
init_coef2 = generate_binary_coef_simple(len(names),1)


# In[533]:


init_coefs = []


# In[534]:


init_coefs += list(init_coef1[:20])
init_coefs += list(init_coef2)


# In[535]:


sequentialmc = SequentialMC(posterior=posterior,pareto_set=ParetoSet(),pareto_filename='pareto_set')


# In[536]:


for i, name in enumerate(names):
    print (i, name)


# In[537]:


#init_coefs[0] = np.array([0,1,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])


# In[514]:


pareto_set = sequentialmc(samples=init_coefs,beta0_nsteps=100,beta0to1_nsteps=1000,beta1_nsteps=1000, mcmc_nsteps=10,feature_descriptions=names )


# In[515]:


prediction = -1.*data[:,0]**3 + 1.9*data[:,0]**2 - 0.9*data[:,0] + 10.*data[:,2]


# In[516]:


pareto_set = sequentialmc.pareto_set


# In[517]:


array = pareto_set.to_array()


# In[518]:


plt.plot(array[:,0], array[:,1],'ro')


# In[519]:


array


# In[520]:


pareto_set[0].data


# In[521]:


pareto_set[1].data


# In[522]:


pareto_set[2].data


# In[523]:


pareto_set[3].data


# In[524]:


pareto_set[4].data


# In[525]:


pareto_set[5].data


# In[526]:


pareto_set[6].data


# In[44]:


pareto_set.save_csv('pareto_set_simulation.csv')

