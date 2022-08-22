import pytest
from src.EKF import ExtendedKalmanFilter
import numpy as np

def test_EKF():
    ekf = ExtendedKalmanFilter()
    x_init = np.array([0.,0.]).reshape(-1,1)
    P_init = np.diag([1e9,1e9])
    ekf.init_state(x_init,P_init)