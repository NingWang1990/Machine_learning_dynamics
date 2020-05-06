import numpy as np
def oneDTravelWave(x,predict_interval=1,gamma=10., m=-0.2, ypsilon=1.,tau=1., delta_x=1,num_time_step_between_frame=1, delta_t_frame=5.):
    """
    x........one d or two d array.
            if it is two d, (#_of_initial_conditions, # of spatial grid points)
            if it is one d, (# of spatial grid points)

    """
    shape0 = x.shape
    if len(shape0) == 1:
        x = np.expand_dims(x,0)
    shape = x.shape
    result = np.zeros((shape[0],shape[1],predict_interval))
    delta_t_integration = delta_t_frame / num_time_step_between_frame
    second_order_deriv = np.array([1., -2., 1.]) / (delta_x*delta_x)
    x_up = x.copy()
    for i in range(predict_interval):
        for j in range(num_time_step_between_frame):
            local_term = -gamma * x_up * (1.- x_up) *(0.5-x_up) + m*x_up*(1-x_up) 
            tt = np.pad(x_up,((0,0),(1,1)),mode='edge')
            shape = tt.shape
            diffuse_term = ypsilon * (tt[:,0:(shape[1]-2)]-2*tt[:,1:(shape[1]-1)] +tt[:,2:(shape[1])] )/(delta_x*delta_x)
            x_up = x_up + (delta_t_integration / tau)*(local_term + diffuse_term)
        result[:,:,i] = x_up
    if len(shape0) == 1:
        result = np.squeeze(result)
    return result

