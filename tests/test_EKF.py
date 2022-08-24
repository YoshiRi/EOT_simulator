from random import sample
from this import d
import pytest
from src.EKF import ExtendedKalmanFilter
import numpy as np

from src.utils import rot_mat_2d

def test_EKF():
    ekf = ExtendedKalmanFilter()
    x_init = np.array([0.,1.]).reshape(-1,1)
    P_init = np.diag([1e9,1e9])
    ekf.init_state(x_init,P_init)


    # prediction test
    def sample_predict(x, y=None, dt=0):
        x_ = np.copy(x)
        x_[0] = x_[1]*dt
        return x_

    Q = np.diag([0.1,0.1])
    dt = 0.1
    x_,P_ =  ekf.predict_nonlinear(sample_predict,Q,None,dt=dt)

    assert x_[0] == ekf.x[0] + ekf.x[1] *dt

    # measurement test 
    def sample_measurement(x):
        z = x[0] * x[1]
        return z

    Rnoise = np.diag([0.02])
    ekf.update_nonlinear(x_, P_, sample_measurement, [0.1], Rnoise)

def test_EKF_case2():
    ekf = ExtendedKalmanFilter()
    x_init = np.array([0.,1.,0.3]).reshape(-1,1)
    P_init = np.diag([1e9]*3)
    ekf.init_state(x_init,P_init)

    dt = 0.1
    A = np.diag([1]*3)
    A[0,1] = 0.1
    Q = np.diag([0.05,0.1,0.01])
    R = np.diag([0.1,0.1])

    def Hfunc(x):
        cta = x[2]
        return rot_mat_2d(cta) @ x[:2]

    z = np.array([1,1]).reshape(-1,1)

    for i in range(100):
        x_ , P_ = ekf.predict_linear(A,Q)
        ekf.update_nonlinear(x_, P_, Hfunc, z, R)

    print(Hfunc(ekf.x),ekf.x, z)
    assert np.allclose(Hfunc(ekf.x), z)