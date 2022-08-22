# -*- coding: utf-8 -*-
"""EKF utility function
"""

from __future__ import (absolute_import, division, unicode_literals)

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from copy import deepcopy
import numpy as np
from src.utils import numerical_jacob, numerical_grad


class ExtendedKalmanFilter():
    def __init__(self) -> None:
        pass

    def init_state(self, xinit, Pinit):
        self.x = xinit
        self.P = Pinit

    def predict_linear(self, A, Q, B=None, u=None, q_u=None):
        """Linear Prediction

        x_{n|n-1} = A x_{n-1} + B u_{n-1} + q(system noise)

        Args:
            A (_type_): state transition 
            Q (_type_): system_noise
            B (_type_, optional): input to state matrix. Defaults to None.
            u (_type_, optional): input . Defaults to None.
            q_u (_type_, optional): input covariance. Defaults to None.

        Returns:
            x_, P_ : Predicted state and covariance
        """
        if B is None:
            x_ = A @ self,x
            P_ = A @ self.P @ self.A.T + Q
        else:
            x_ = A @ self.x + B @ u
            P_ = A @ self.P @ self.A.T + Q + B @ q_u @ B.T
        return x_, P_
        

    def predict_nonlinear(self, F_func, Q, Jacob_F = None, *other_arg):
        """General Prediction

        Args:
            F_func (_type_): state transition function
            Q : System noise matrix len(x)*len(x)
            Jacob_F (_type_, optional): Function returns Jacobian matrix . Defaults to None.
            other_arg: other args

        Returns:
            x_, P_ : Predicted state and covariance
        """

        if Jacob_F is None:
            J_F = numerical_grad(F_func, self.x, *other_arg)# numerical grad to calc jacobian
        else:
            J_F = Jacob_F(self.x, *other_arg)

        x_ = F_func(self.x, *other_arg)
        P_ = J_F @ self.P @ J_F.T + Q
        return x_, P_

    def update_linear(self, x_, P_, C, z, R):
        """Linear Update

        Args:
            x_ (_type_): predicted state
            P_ (_type_): predicted cov
            C (_type_): measurement matrix
            z (_type_): measurement vector
            R (_type_): measurement cov
        """
        err = z - C @ x_
        S = C @ P_ @ C.T + R
        K = P_ @ C.T @ np.linalg.inv(S)
        self.x = x_ + K @ err
        self.P = (np.eye(P_.shape[0]) - K @ C) @ P_

    def update_nonlinear(self, x_, P_, H_func, z, R, H_jacob=None, *other_args):
        """Non Linear Update

        Args:
            x_ (_type_): predicted state
            P_ (_type_): predicted cov
            H_func (_type_): measurement func
            z (_type_): measurement vector
            R (_type_): measurement cov
            H_jacob (_type_, optional): . Defaults to None.
            other_arg (_type_): other arg
        """


        if H_jacob is None:
            J_H = numerical_jacob(H_func,x_,*other_args)
        else:
            J_H = H_jacob(x_, *other_args)
        
        err = z - H_func(x_, *other_args)
        S = J_H @ P_ @ J_H.T + R
        K = P_ @ J_H.T @ np.linalg.inv(S)
        self.x = x_ + K @ err
        self.P = (np.eye(P_.shape[0]) - K @ J_H) @ P_



