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
    

def test_ellipsoid(showflag=False):
    shape = EllipsoidData()
    shape.init_with_param([0,0],10,8,30)
    if showflag:
        import matplotlib.pyplot as plt
        plt.figure()
        shape.plot()
        plt.xlim([-20,20])
        plt.ylim([-20,20])
        plt.grid()
    
    Cov=shape.calc_ellipse_matrix(100,64,np.deg2rad(30))
    shape.init_with_cov([0,0],Cov)
    if showflag:
        plt.figure()
        shape.plot()
        plt.xlim([-20,20])
        plt.ylim([-20,20])
        plt.grid()
        plt.show()


def test_rectangle(showflag=False):
    shape = RectangleData()
    shape.center = [0,0]
    shape.width = 1
    shape.length = 2
    shape.orientation = 0.2

    if showflag:
        plt.figure(1)
        plt.xlim([-20,20])
        plt.ylim([-20,20])
        shape.plot()
        plt.show()
