import numpy as np

def rot_mat_2d(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([[c,-s],[s,c]]).reshape(2,2)