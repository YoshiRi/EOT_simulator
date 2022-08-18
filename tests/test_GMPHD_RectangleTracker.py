import pytest
from src.tracker.GMPHD_RectangleTracker import *

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
    assert find_corner_index(z_1) == 1