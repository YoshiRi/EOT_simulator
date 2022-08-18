import pytest
from src.utils import * 

def test_rot_mat_2d():
    vec = np.array([3,4]).reshape(-1,1)
    assert np.array_equal(vec,rot_mat_2d(0) @ vec)