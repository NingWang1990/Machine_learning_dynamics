import numpy as np

def ChebyshevLocalFit_2D(u, x, y, t, deg=3, diff=1, width=5):
    """
    u.......... ndarray of shape (n_x, n_y, n_t), values of the function
    x.......... ndarray of shape (n_x,), x-coordinates where values are known
    y.......... ndarray of shape (n_y,), y-coordinates where values are known 
    t...........ndarray of shape(n_t,), temporal grid points 
    deg........ degree of polynomial to use
    diff....... maximum order of derivatives we want
    width.......width of window to fit to polynomial
    """
    u = np.array(u)
    x = np.array(x)
    y = np.array(y)
    if not u.ndim == 3:
        raise ValueError('u must be a 3D array')
    if not x.ndim == 1:
        raise ValueError('x must be a 1D array')
    if not y.ndim == 1:
        raise ValueError('y must be a 1D array')
    if not t.ndim == 1:
        raise ValueError('t must be a 1D array')
    if not u.shape[0] == len(x):
        raise ValueError('length of x must be identical to the size of the first dimension of u')
    if not u.shape[1] == len(y):
        raise ValueError('length of y must be identical to the size of the second dimension of u')
    if not u.shape[1] == len(t):
        raise ValueError('length of t must be identical to the size of the third dimension of u')
    if (not diff == 1 ) and (not diff == 2):
        raise valueError('diff must be 1 or 2')

    nx = len(x)
    ny = len(y)
    nt = len(t)
    du_x = np.zeros((nx-2*width, ny-2*width, nt-2*width))
    du_y = np.zeros((nx-2*width, ny-2*width, nt-2*width))
    du_t = np.zeros((nx-2*width, ny-2*width, nt-2*width))
    if diff == 2:
        du_xx = np.zeros((nx-2*width, ny-2*width, nt-2*width))
        du_yy = np.zeros((nx-2*width, ny-2*width, nt-2*width))
        du_xy = np.zeros((nx-2*width, ny-2*width, nt-2*width))

    for i in range(width, nx-width):
        for j in range(width, ny-width):
            for k in range(width, nt-width):
                # x
                points = np.arange(i-width, i+width)
                poly = np.polynomial.chebyshev.Chebyshev.fit(x[points],u[points,j,k], deg)
                du_x[i-width, j-width, k-width] = poly.deriv(m=1)(x[i])
                if diff == 2:
                    du_xx[i-width, j-width, k-width] = poly.deriv(m=2)(x[i])
                # y 
                points = np.arange(j-width, j+width)
                poly = np.polynomial.chebyshev.Chebyshev.fit(y[points],u[i,points,k], deg)
                du_y[i-width, j-width, k-width] = poly.deriv(m=1)(y[j])
                if diff == 2:
                    du_yy[i-width, j-width, k-width] = poly.deriv(m=2)(y[j])
                

                # t 
                points = np.arange(k-width, k+width)
                poly = np.polynomial.chebyshev.Chebyshev.fit(t[points],u[i,j,points], deg)
                du_t[i-width, j-width, k-width] = poly.deriv(m=1)(t[k])
                
    # use a simple finite difference method to calculate du_xy
    if diff == 2:
        du_xy = 0.5 * (np.gradient(du_y,[x[width:(nx-width)], y[width:(ny-width)], t[width:(nt-width)]], axis=0) + \
                   np.gradient(du_x,[x[width:(nx-width)], y[width:(ny-width)], t[width:(nt-width)]], axis=1) )   
                
    if diff == 1:
        return (du_x, du_y, du_t)
    elif diff == 2:
        return (du_x, du_y, du_xx, du_yy, du_xy, du_t)

def ChebyshevLocalFit_1D(u, x, t, deg=3, diff=1, width=5):
    
    """
    This function is adapted from https://github.com/snagcliffs/PDE-FIND
    fit a Chebyshev polinomial locally, and use it to calculate derivatives.
    u.......... ndarray of shape (n_x, n_t), values of some function
    x.......... ndarray of shape (n_x,),x-coordinates where values are known
    t.......... ndarray of shape (n_t), temporal grid points where values are known
    deg........ degree of polynomial to use
    diff....... maximum order of derivatives we want
    width.......width of window to fit to polynomial
    """
    u = np.array(u)
    x = np.array(x)
    t = np.array(t)
    if (not u.ndim == 2):
        raise ValueError('u must be 2D array')
    if (not x.ndim == 1):
        raise ValueError('x must be 1D array')
    if (not t.ndim == 1):
        raise ValueError('t must be 1D array')
    if not u.shape[0] == len(x):
        raise ValueError('lengths of x must be identical to number of rows in u')
    if not u.shape[1] == len(t):
        raise ValueError('length of t must be identical to number of columns in u')
    if (not diff == 1) and (not diff == 2):
        raise ValueError('diff must be 1 or 2')

    nx = len(x)
    nt = len(t)

    # Take the derivatives in the center of the domain
    du_x = np.zeros((nx-2*width, nt-2*width))
    du_t = np.zeros((nx-2*width, nt-2*width))
    if diff == 2:
        du_xx = np.zeros((nx-2*width, nt-2*width))
    for i in range(width, nx-width):
        for j in range(width, nt-width):
            # x
            points = np.arange(i-width, i+width)
            poly = np.polynomial.chebyshev.Chebyshev.fit(x[points],u[points,j],deg)
            du_x[i-width, j-width] = poly.deriv(m=1)(x[i])
            if diff == 2:
                du_xx[i-width, j-width] = poly.deriv(m=2)(x[i])
            # t
            points = np.arange(j-width, j+width)
            poly = np.polynomial.chebyshev.Chebyshev.fit(t[points],u[i,points],deg)
            du_t[i-width, j-width] = poly.deriv(m=1)(t[j])
    
    if diff == 1:
        return (du_x, du_t)
    elif diff == 2:
        return (du_x, du_xx, du_t)
