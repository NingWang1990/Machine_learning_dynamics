import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import sys
sys.path.append("..")
from util.oneDTravelWave import oneDTravelWave
from scipy.optimize import least_squares
from scipy.io import loadmat
import matplotlib as mpl

def residual(paras, X, Y,tau, delta_x,num_time_step_between_frame, delta_t_frame):
    """
    paras[0]: ypsilon
    paras[1]: m
    paras[2]: gamma
    """
    ypsilon = paras[0]
    gamma = paras[1]
    m = paras[2]
    shape = Y.shape
    predict_interval = shape[2]
    prediction = oneDTravelWave(X, predict_interval,gamma, m, ypsilon,tau, delta_x,num_time_step_between_frame,delta_t_frame)
    #error = np.mean(np.square(Y-prediction))
    return np.reshape(Y-prediction,(-1))
    
def get_data(phase_fields, ids, predict_interval):
    shape = phase_fields.shape
    X = np.zeros((len(ids),shape[0]))
    Y = np.zeros((len(ids),shape[0],predict_interval))
    for i,ii in enumerate(ids):
        X[i,:]= phase_fields[:,ii]
        Y[i,:,:] = phase_fields[:,(ii+1):(ii+1+predict_interval)]
    #return np.array(X), np.swapaxes(np.array(Y),1,2)
    return X, Y

class fittingOneDTravelWave():
    def __init__(self,phase_fields, x, t,num_time_step_between_frame=1):
        """
        phase_fields.....two D array, (space, time)
        x................one D array, space
        t................one D array, time
        num_time_step_between_frame......init, split the time step between frames into a smaller one in order to guarantee a good numerial stability
       """
        self.phase_fields = phase_fields
        self.x = x
        self.t = t
        self.num_time_step_between_frame = num_time_step_between_frame
        shape = phase_fields.shape
        if not shape[0] == len(x):
            raise ValueError()
        if not shape[1] == len(t):
            raise ValueError()
        self.opt = None

    def perform_fitting(self,init_values=[20., 1., -0.4],predict_interval=1,tau=1.):
        """
        init_values......................initial values [ypsilon, gamma, m]
        predict_interval.................init, use multiple frames as the training output.
        tau..............................float, relaxation time
        """
        delta_x = self.x[1]-self.x[0]      
        delta_t_frame = self.t[1] - self.t[0]
        shape = self.phase_fields.shape
        ids = np.arange(shape[1]-predict_interval)
        X,Y = get_data(self.phase_fields,ids,predict_interval)        
        opt = least_squares(residual,init_values,args=[X, Y, tau,delta_x, self.num_time_step_between_frame, delta_t_frame], ftol=1e-10, xtol=1e-10,)
        self.opt = opt
        return self.opt.x
    def prediction(self):
        if self.opt == None:
            raise ValueError('first perform fitting')
        data = self.phase_fields.copy()
        data[:,1:] = oneDTravelWave(self.phase_fields[:,0], predict_interval=len(self.phase_fields[0])-1,ypsilon=self.opt.x[0], gamma=self.opt.x[1],
                                    m=self.opt.x[2], num_time_step_between_frame=self.num_time_step_between_frame, delta_t_frame=self.t[1]-self.t[0],delta_x=self.x[1]-self.x[0])
        return data
