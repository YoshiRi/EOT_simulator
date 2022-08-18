"""
Rectangle Shape Tracker based on GM-PHD filter


http://liu.diva-portal.org/smash/get/diva2:434601/FULLTEXT02.pdf


Yoshi Ri
2022/08/18
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")


import matplotlib.pyplot as plt
import numpy as np
import itertools
from enum import Enum

from utils import rot_mat_2d






def estimate_number_of_sides(measurements,threshold=25):
    """Estimate number of measured rectangle sides

    Args:
        measurements (_type_): list of measured points coorinate
        threshold (int, optional): _description_. Defaults to 25.

    Returns:
        _type_: _description_
    """
    Nz = len(measurements)
    if Nz == 1:
        N = 1
    else:
        Z = np.array(measurements).reshape(Nz,2) # to numpy array
        Zbar = np.mean(Z,axis=0)
        Znorm = Z-Zbar
        C = Znorm.T @ Znorm
        C = C/(Nz-1)
        assert C.shape == (2,2), "Real shape is " + str(C.shape)
        eigs = sorted( np.linalg.eigvals(C) )
        
        if eigs[0] == 0: # prevent zero division
            N = 1
        elif eigs[1]/eigs[0] > threshold:
            N = 1
        else:
            N = 2

    return N


def point2line_distance(z1,z2,z3):
    """return distance between point and line

    Args:
        z1 (list): point of line edge
        z2 (list): point of line edge
        z3 (list): target point to calc distance

    Returns:
        double : distance between z3 and the line passing z1 and z2
    """
    Num = np.sqrt( (z2[0]-z1[0])**2 + (z2[1]-z1[1])**2 )
    Den = (z2[0]-z1[0])*(z1[1]-z3[1]) + (z1[0]-z3[0])*(z2[1]-z1[1])
    d = np.abs(Den/Num)
    return d


def find_corner_index(measurements):
    """Extract corner measurements

    Args:
        measurements (list of list): List of 2D measurements 
    Returns:
        _type_: iter
    """
    Nz = len(measurements)

    dmin = 1e5*Nz
    corner_index = None

    if Nz < 3:
        return corner_index

    for n in range(1,Nz-1):
        d = 0
        for k in range(1,n):
            d += point2line_distance(measurements[0],measurements[n],measurements[k])
        for k in range(n+1,Nz-1):
            d += point2line_distance(measurements[n],measurements[-1],measurements[k])
        
        if d < dmin:
            corner_index = n
            dmin = d
    return corner_index

