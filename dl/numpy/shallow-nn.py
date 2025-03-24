import numpy as np 


def init_params(n_x,n_h,y):
    np.random.seed(1) 
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h,1))
    