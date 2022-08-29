import pytest
from src.tracker.RectangleTracker import *


def test_get_estimated_rectangular_points():
    Z = [[0,1],[0,0],[1,0],[2,0],[3,0]]
    state = np.array([0,0,0,0,0,1,0.5]).reshape(-1,1)
    get_estimated_rectangular_points(state,Z)


def test_measurement_normalvec_angle():
    Z1 = [[0,0],[0,1]]
    Z2 = [[0,-1],[0,-1]]
    assert measurements_normalvec_angle(Z1) == np.pi
    assert measurements_normalvec_angle(Z2) == np.pi/2.0
    
def test_estimate_number_of_sides():
    z_1 = [[0,0],[1,0],[2,0],[3,0]]
    z_2 = [[0,1],[0,0],[1,0],[2,0],[3,0]]

    assert estimate_number_of_sides(z_1) == 1
    assert estimate_number_of_sides(z_2) == 2

def test_point2line_distance():
    z1 = [0,0]
    z2 = [1,0]
    z3 = [2,6]
    assert point2line_distance(z1,z2,z3) == 6

def test_find_corner_index():
    z_1 = [[0,1],[0,0],[1,0],[2,0],[3,0]]
    assert find_corner_index(z_1)[0] == 1

    z_2 = [[0,1],[0,0],[1,0]]
    assert find_corner_index(z_2)[0] == 1


def test_coords_divide():
    rsp = RectangleShapePrediction()
    rsp.center = [0,0]
    rsp.width = 4
    rsp.length = 10

    front = rsp.divide_coords(1,0)[0]
    estimated_front = np.array([0,rsp.length/2])
    assert np.allclose(front, estimated_front)

    right = rsp.divide_coords(3,1)[1]         # take center point
    estimated_right = np.array([rsp.width/2,0])
    assert np.allclose(right, estimated_right) # array_equal sometimes fail due to calc error


def test_Bicycle_model():
    bmm = BicycleMotionModel()
    x = np.array([0.]*7)
    x[5] = 2
    x[6] = 1
    x_ = bmm.predict(x, dt=0.1)
    x[3] = 1
    x[4] = 0.5
    x_ = bmm.predict(x, dt=0.1)
    

def test_Simple_model():
    bmm = ConstantVelocityModel()
    x = np.array([0.]*7)
    x[5] = 2
    x[6] = 1
    x_ = bmm.predict(x, dt=0.1)
