from random import sample
from this import d
import pytest
from src.EKF import ExtendedKalmanFilter
import numpy as np

def test_EKF():
    ekf = ExtendedKalmanFilter()
    x_init = np.array([0.,1.]).reshape(-1,1)
    P_init = np.diag([1e9,1e9])
    ekf.init_state(x_init,P_init)

    def sample_predict(x, y=None, dt=0):
        x_ = np.copy(x)
        x_[0] = x_[1]*dt
        return x_

    Q = np.diag([0.1,0.1])
    dt = 0.1
    x_,P_ =  ekf.predict_nonlinear(sample_predict,Q,None,dt=dt)

    assert x_[0] == ekf.x[0] + ekf.x[1] *dt
    