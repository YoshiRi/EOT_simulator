import pytest
from src.utils import * 

def test_rot_mat_2d():
    vec = np.array([3,4]).reshape(-1,1)
    assert np.array_equal(vec,rot_mat_2d(0) @ vec)

def test_rad_distance():
    assert rad_distance(0,2*np.pi) == 0
    assert rad_distance(0,-5*np.pi) == np.pi


def test_numerical_grad():
    x_square = lambda x: x*x
    x0, x1, x2 = np.array([0.0]), np.array([1.0]), np.array([1e9])
    assert numerical_grad(x_square, x0, 0) == 0
    assert np.allclose(numerical_grad(x_square, x1, 0) , 2.0*x1)
    assert numerical_grad(x_square, x2, 0) ==  2.0*x2 

    x_vec = lambda x: np.array([np.sin(x), np.cos(x)])
    x_vec_a = lambda x: np.array([np.cos(x), -np.sin(x)])
    assert np.allclose(numerical_grad(x_vec,x1,0), x_vec_a(x1))

def test_numerical_jacob():
    func = lambda x: x[0]*x[0] + np.cos(x[1])
    d_func = lambda x: np.array( [2*x[0],-np.sin(x[1])]).reshape(1,-1)
    X0 = np.array([1.0,0.5]).reshape(-1,1)

    J = numerical_jacob(func,X0)
    assert np.allclose(J,d_func(X0)) 
    