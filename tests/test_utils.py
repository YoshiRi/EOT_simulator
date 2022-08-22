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
    