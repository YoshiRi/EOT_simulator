import numpy as np

def rot_mat_2d(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([[c,-s],[s,c]]).reshape(2,2)

def vec2rad(v):
    return np.arctan2(v[1],v[0])

def rad_distance(a1,a2):
    dist = np.abs(a1 - a2)
    while dist >= 2*np.pi:
        dist -= 2*np.pi
    return dist
