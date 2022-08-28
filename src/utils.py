from os import stat_result
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
    if np.abs(dist) > np.pi:
        min_dist  = 2*np.pi - np.abs(dist)  
    return min_dist


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
        grad = grad_.reshape(-1,1)
        Jacob = np.hstack([Jacob,grad]) if Jacob.size else grad
    return Jacob


# Shape Objects Temporary put here
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle


class RectangleData:

    def __init__(self, plot_setting="-.r"):
        self.center = [None]*2
        self.width = None
        self.length = None
        self.orientation = None
        self.plot_setting = plot_setting

    def calc_contour(self):
        """Return rectangle contour"""
        R = rot_mat_2d(self.orientation) 
        center = np.array(self.center).reshape(-1,1)
        v1 = np.array([self.width/2, self.length/2]).reshape(-1,1)
        v2 = np.array([self.width/2, -self.length/2]).reshape(-1,1)
        c1 = R @ v1 + center
        c2 = R @ v2 + center
        c3 = -R @ v1 + center
        c4 = -R @ v2 + center
        contour = np.hstack([c1,c2,c3,c4,c1]).T
        return contour

    def draw_rect(self):
        """draw rectangle contour and center without rectangle patch"""
        contour = self.calc_contour()
        plt.plot(contour[:,0],contour[:,1],self.plot_setting)
        plt.plot(self.center[0],self.center[1],'ko')


    def plot(self):
        ax = plt.gca()
        self.draw_rect()



class EllipsoidData:

    def __init__(self):
        self.center = [None] * 2
        self.width = None
        self.height = None
        self.angle_deg = None

    def init_with_param(self,center, l1, l2, angle_deg):
        self.center = center
        self.width, self.height = 2*l1, 2*l2
        self.angle_deg = angle_deg

    def init_with_cov(self, center, cov):
        self.center = center
        w2,h2,theta = self.decompose_ellipse_matrix(cov)
        self.width = 2*np.sqrt(w2)
        self.height = 2*np.sqrt(h2)
        self.angle_deg = np.rad2deg(theta)

    def plot(self):
        patch = Ellipse(self.center, self.width, self.height, self.angle_deg,
                        fill=False, ls='--')
        plt.gca().add_patch(patch)
        plt.plot(*self.center,'*')

    @staticmethod
    def calc_ellipse_matrix(l1,l2,theta):
        R = rot_mat_2d(theta)
        V = np.diag([l1,l2])
        return R @ V @ R.T

    @staticmethod
    def decompose_ellipse_matrix(X):
        eigenValues,eigenVectors = np.linalg.eig(X)
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        theta = -np.arctan2(eigenVectors[0,1], eigenVectors[0,0])
        return eigenValues[0], eigenValues[1], theta


