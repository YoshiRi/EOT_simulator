import numpy as np
from copy import deepcopy

def rot_mat_2d(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([[c,-s],[s,c]]).reshape(2,2)

def vec2rad(v):
    return np.arctan2(v[1],v[0])

def rad_distance(a1,a2):
    dist = np.abs(a1 - a2)
    while dist >= 2*np.pi:
        dist -= 2*np.pi
    return dist


# numerical gradient input: func must be a function and x is vector 
def numerical_grad(func, x, i, eps=1e-9, **other_args):
    x_arr = np.array(x,dtype=float).reshape(-1)
    x_p, x_m = deepcopy(x_arr), deepcopy(x_arr)
    r_eps = max(np.abs(x_m[i])*eps, eps)
    x_m[i] -= r_eps
    x_p[i] += r_eps
    fp = func(x_p, **other_args)
    fm = func(x_m, **other_args)
    grad = (fp-fm)/r_eps/2
    return grad

def numerical_jacob(func, x, **other_args):
    x = np.array(x).reshape(-1)
    N = len(x)
    Jacob = np.array([])
    for i in range(N):
        grad_ = numerical_grad(func,x,i,**other_args)
        grad = grad_.reshape(1,-1)
        Jacob = np.vstack([Jacob,grad]) if Jacob.size else grad
    return Jacob
