"""Random Matrix Tracker with ellipsoid surface

Shape: Ellipsoid
Observation model: Surface
Shape tracking method: Random Matrix

Followed Paper: 
- [Basic Concept] M. Feldmann, D. Franken, and J. W. Koch, “Tracking of extended objects and group targets using random matrices,” IEEE Transactionson Signal Processing, vol. 59, no. 4, pp. 1409–1420, Apr. 2011
- [Overview] Granström, Karl; Baum, Marcus (2022): A Tutorial on Multiple Extended Object Tracking. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.19115858.v1
- [ ] 

Yoshi Ri
yoshi.ri@tier4.jp
2022/08/19
"""


from pyexpat.errors import XML_ERROR_CANT_CHANGE_FEATURE_ONCE_PARSING
from re import X
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy
import numpy as np
import itertools
from enum import Enum

from simulator import PerceptionSimulator, VehicleSimulator, LidarSimulator

LIDAR_NOISE_SIGMA = 0.01



class RandomMatrixTracker():
    """Single Object Tracker with Ellipsoid assumption
    """
    def __init__(self):
        """m: physical state
        """
        self.initialized = False
        self.m = np.array([0,0,0,0]).reshape(-1,1)
        self.H = np.array([1,0,0,0, 0,1,0,0]).reshape(2,4)
        self.P = np.diag([1e9,1e9,1e9,1e9])
        self.v = 8 # initial value
        self.V = np.diag([1000,1000])
        self.zscaler = 1.0/4.0 # scale value
        self.log = []

        self.R = np.diag([LIDAR_NOISE_SIGMA,LIDAR_NOISE_SIGMA]) # sensor noise



    def predict_and_update(self,ox,oy,dt,tau=2,process_noise=10):
        """Do Prediction and Update with sensor input

        Args:
            ox (_type_): x coordinate of lidar observation
            oy (_type_): y coordinate of lidar observation
            dt (_type_): sampling tme
            tau (int, optional): tuning parameter for shape prediction agility. larger tau make shape estimation faster. Defaults to 2.
            process_noise (int, optional): Random acceleration for object [m/s/s].  Defaults to 10.

        Returns:
            _type_: _description_
        """
        self.tau_predict = tau
        self.q_acc = process_noise
        
        assert len(ox)==len(oy), "Measurement must have same number of x and y axis value."
        obsnum = len(ox)
        if obsnum==0:
            print("No obs!")
            self.m, self.P, self.v, self.V = self.predict(dt)
            return self.return_shape_and_ids(obsnum)
        m_,P_,v_,V_=self.predict(dt)

        z = [ox,oy]
        self.m, self.P, self.v, self.V = self.measurement_update(m_,P_,v_,V_,z)
        return self.return_shape_and_ids(obsnum)

    def predict(self,dt):
        """Prediction Step

        Args:
            dt (_type_): sampling time
        
        Returns:
            _type_: position state, position cov, shape scalar, shape matrix
        
        Note:

        m_ = F m
        P_ = F P F^T + Q
        v_ = 2d + 2 + e−Ts/τ (v − 2d − 2)
        V_ = v+−2d−2 / v−2d−2 V

        """
        # matrix assume state shape is [x, y, vx, vy]
        A_ = [[1,0,dt,0],
            [0,1,0,dt],
            [0,0,1,0],
            [0,0,0,1]]
        A = np.array(A_).reshape(4,4)
        Bx = np.array([dt*dt/2,0, dt,0]).reshape(-1,1)
        By = np.array([0,dt*dt/2,0, dt]).reshape(-1,1)
        omega = np.array([self.q_acc]).reshape(-1,1) # system noise covariance: acc

        d = 2 # measurement dimension
        m_ = A @ self.m 
        P_ = A @ self.P @ A.T  + Bx @ omega @ Bx.T + By @ omega @ By.T # additive system noise
        v_ = 2*d + 2 + np.exp(-dt/self.tau_predict) *(self.v - 2*d -2)
        V_ = (v_ - 2*d -2)/(self.v - 2*d -2)*self.V
        return m_, P_, v_, V_

    def measurement_update(self,m_,P_,v_,V_,z):
        """Update
        """ 
        n = len(z[0]) # measurement num
        dim = 2 # dimention
        z_bar, Z = self.calc_measurement_mean_cov(z)
        err = z_bar - self.H @ m_
        X_hat = V_/(v_ - 2*dim - 2)
        Y = self.zscaler*X_hat + self.R
        S = self.H @ P_ @ self.H.T  +  Y/n
        S_inv = np.linalg.inv(S)
        K = P_ @ self.H.T @ S_inv # kalman gain
        X_hat_sqrt = scipy.linalg.sqrtm(X_hat)
        S_inv_sqrt = scipy.linalg.sqrtm(S_inv)
        Y_inv_sqrt = scipy.linalg.sqrtm(np.linalg.inv(Y))
        N_hat = X_hat_sqrt @ S_inv_sqrt @ err @ err.T @ S_inv_sqrt.T @ X_hat_sqrt.T
        Z_hat = X_hat_sqrt @ Y_inv_sqrt @ Z @ Y_inv_sqrt.T @ X_hat_sqrt.T
        # update
        m = m_ + K @ err
        P = P_ - K @ S @ K.T
        v = v_ + n
        V = V_ + N_hat + Z_hat
        #print(K @ err)
        #print(P)


        return m, P, v, V

    def calc_measurement_mean_cov(self,z):
        z_bar = np.mean(z,axis=1) # assume z is list of [ox, oy]
        Z_ = np.array(z).T - z_bar
        z_cov = Z_.T @ Z_
        return z_bar.reshape(-1,1), z_cov

    def return_shape_and_ids(self,obsnum):
        shape = EllipsoidData()
        X = self.V/(self.v - 2*2 - 2)
        pos = self.H @ self.m
        shape.init_with_cov(pos,X)
        ids = np.arange(0,obsnum)
        return [shape], [ids]

    def fitting(self,ox,oy):
        dt = 0.1
        out = self.predict_and_update(ox,oy,dt)
        self.log.append([self.m, self.P, self.v, self.V])
        return out 

    def __del__(self):
        plt.figure(2)
        x = [log[0][0] for log in self.log]
        t = np.arange(len(x))
        y = [log[0][1] for log in self.log]
        vx = [log[0][2] for log in self.log]
        vy = [log[0][3] for log in self.log]
        plt.plot(t,x,t,y,t,vx,t,vy)
        plt.legend(["x","y","vx","vy"])
        plt.figure(3)
        vdata = [log[2] for log in self.log]
        plt.plot(vdata)
        plt.figure(4)
        pdata1 = [log[1][0,0] for log in self.log]
        pdata2 = [log[1][1,1] for log in self.log]
        plt.plot(pdata1)
        plt.plot(pdata2)
        plt.show()


# Util
def rot_mat_2d(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([[c,-s],[s,c]])

def calc_ellipse_matrix(l1,l2,theta):
    R = rot_mat_2d(theta)
    V = np.diag([l1,l2])
    return R @ V @ R.T


""" OLD
def decompose_ellipse_matrix(X):
    s = X[0,0] + X[1,1]
    p = X[0,0] + X[1,1] - X[0,1]*X[0,1]
    l1 = 0.5 * (s + np.sqrt(s*s - 4*p))
    l2 = 0.5 * (s - np.sqrt(s*s - 4*p))
    sin_2a = 2*X[0,1]/(l1-l2)
    theta = np.arcsin(sin_2a)/2
    return l1, l2, theta
"""

def decompose_ellipse_matrix(X):
    eigenValues,eigenVectors = np.linalg.eig(X)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    theta = -np.arctan2(eigenVectors[0,1], eigenVectors[0,0])
    return eigenValues[0], eigenValues[1], theta

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
        w2,h2,theta = decompose_ellipse_matrix(cov)
        self.width = 2*np.sqrt(w2)
        self.height = 2*np.sqrt(h2)
        self.angle_deg = np.rad2deg(theta)

    def plot(self):
        patch = Ellipse(self.center, self.width, self.height, self.angle_deg,
                        fill=False, ls='--')
        plt.gca().add_patch(patch)
        plt.plot(*self.center,'*')

def test_ellipsoid():
    shape = EllipsoidData()
    shape.init_with_param([0,0],10,8,30)
    plt.figure()
    shape.plot()
    plt.xlim([-20,20])
    plt.ylim([-20,20])
    plt.grid()
    
    Cov=calc_ellipse_matrix(100,64,np.deg2rad(30))
    shape.init_with_cov([0,0],Cov)
    plt.figure()
    shape.plot()
    plt.xlim([-20,20])
    plt.ylim([-20,20])
    plt.grid()
    plt.show()


def senario1():
    sim = PerceptionSimulator(dt=0.1)
    v1 = VehicleSimulator(-10.0, 0.0, np.deg2rad(90.0),
                          0.0, 50.0 / 3.6, 3.0, 5.0)

    sim.append_vehicle(v1)

    tracker = RandomMatrixTracker()

    sim.run(tracker)
    print("Done")


if __name__=="__main__":
    #test_ellipsoid()
    senario1()