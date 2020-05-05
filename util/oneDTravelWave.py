import numpy as np
def oneDTravelWave(x,predict_interval,gamma, mass, ypsilon,tao=1, delta_x=1,delta_t_integration=0.1, delta_t_frame=5):
    """
    x........2D array
    """
    shape = x.shape
    result = np.zeros((shape[0],shape[1],predict_interval))
    second_order_deriv = np.array([1., -2., 1.]) / (delta_x*delta_x)
    x_up = x.copy()
    for i in range(predict_interval):
        num_frames = int(delta_t_frame/delta_t_integration)
        for j in range(num_frames):
            local_term = -gamma * x_up * (1.- x_up) *(0.5-x_up) + mass*x_up*(1-x_up) 
            tt = np.pad(x_up,((0,0),(1,1)),mode='edge')
            shape = tt.shape
            diffuse_term = ypsilon * (tt[:,0:(shape[1]-2)]-2*tt[:,1:(shape[1]-1)] +tt[:,2:(shape[1])] )/(delta_x*delta_x)
            x_up = x_up + (delta_t_integration / tao)*(local_term + diffuse_term)
        result[:,:,i] = x_up
    return result

