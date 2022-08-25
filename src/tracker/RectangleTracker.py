"""Basic functions Rectangle Tracker Utility
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import logging

import numpy as np
from utils import rot_mat_2d, vec2rad, rad_distance


# temporary 
def get_estimated_rectangular_points(state,measurements):
    Z = np.array(measurements).reshape(-1,2)
    side_num = estimate_number_of_sides(Z)
    rs_obj = init_rectangle(state)
    estimated_measurements = np.array([])

    if side_num == 1:
        nvec_rad = measurements_normalvec_angle(Z)
        idx = rs_obj.find_closest_angle(nvec_rad)
        estimated_measurements = rs_obj.divide_coords(Z.shape[0],idx)

    elif side_num == 2:
        corner_index, parted_measurements = find_corner_index(Z)
        for pm in parted_measurements:
            nvec_rad = measurements_normalvec_angle(pm)
            idx = rs_obj.find_closest_angle(nvec_rad,2) # this could be None when nvec_rad is not good
            em = rs_obj.divide_coords(pm.shape[0],idx)
            estimated_measurements = np.vstack([estimated_measurements,em]) if estimated_measurements.size else em
        
        # When N = 3 (corner is duplicated)
        if estimated_measurements.shape[0] == Z.shape[0]+1:
            estimated_measurements = np.delete(estimated_measurements,corner_index, axis=0) # remove duplicated corner

    return estimated_measurements.reshape(-1,1)



def init_rectangle(state):
    """
    Set Rectangle Shape information from state [x, y, v, psi, theta, l ,w]
    """
    rs = RectangleShapePrediction()
    state = state.reshape(-1)
    rs.orientation = state[3]
    rs.center = np.array([state[0], state[1]]).reshape(-1,1)
    rs.length = state[5]
    rs.width = state[6]
    return rs

def measurements_normalvec_angle(measurements):
    """calc normal vector angle of the 

    Args:
        measurements (_type_): measurements [[z1x, z1y],...,[znx,zny]]

    Returns:
        _type_: angle of z1 -> zn + pi/2
    """
    assert len(measurements) > 1, "measurements num must be larger than 1"
    Z = np.array(measurements).reshape(-1,2)
    p0 = Z[0] # start
    pn = Z[1] # end
    dp = [pn[i] - p0[i] for i in range(2)]
    return vec2rad(dp) + np.pi/2.0

def estimate_number_of_sides(measurements,threshold=25):
    """Estimate number of measured rectangle sides

    Args:
        measurements (_type_): list of list of measured points coorinate
        threshold (int, optional): _description_. Defaults to 25.

    Returns:
        _type_: _description_
    """
    Z = np.array(measurements).reshape(-1,2)
    Nz = Z.shape[0]

    if Nz == 1:
        N = 1
    else:
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

def safe_mean(sum,num):
    if num > 0:
        return sum/num
    else:
        return sum

def find_corner_index(measurements):
    """Extract corner measurements

    Args:
        measurements (list of list): List of 2D measurements 
    Returns:
        corner_index : corner index 
        divided indexes: 
    """
    measurements = np.array(measurements).reshape(-1,2)
    Nz = measurements.shape[0]

    dmin = 1e5*Nz
    corner_index = None

    assert Nz> 2, "point num must be larger than 2!"

    for n in range(1,Nz-1):
        d, d_a, d_b = 0, 0, 0
        for k in range(1,n):
            d_a += point2line_distance(measurements[0],measurements[n],measurements[k])
        for k in range(n+1,Nz-1):
            d_b += point2line_distance(measurements[n],measurements[-1],measurements[k])
        d = d_a + d_b

        if d < dmin:
            corner_index = n
            dmin = d
            d_a_mean = safe_mean(d_a,len(range(1,n)))
            d_b_mean = safe_mean(d_b,len(range(n+1,Nz-1)))
            
    if d_a_mean < d_b_mean:
        parted_measurements = [measurements[:corner_index+1], measurements[corner_index+1:]]
    elif d_a_mean > d_b_mean:
        parted_measurements = [measurements[:corner_index], measurements[corner_index:]]
    else:
        parted_measurements = [measurements[:corner_index+1], measurements[corner_index:]] # this is happened when N = 3
    return corner_index, parted_measurements


class RectangleShapePrediction():
    def __init__(self) -> None:
        self.orientation = None
        self.center = None
        self.width = None
        self.length = None

    def normal_vector_angles(self):
        """Return Normal vector angles

        front

        +--b1---+
        |       |
        |       |
        b4  ^   b2
        |   |   |
        |       |
        |       |
        +---b3--+

        Returns:
            list of normal vector angle of  [b1, b2, b3, b4] (outside)
        """
        n_vecs = [self.orientation , self.orientation - np.pi/2, self.orientation + np.pi, self.orientation + np.pi/2]
        return n_vecs
    
    def find_closest_angle(self,angle_rad, threshold=0.5):
        """find closest direction

        Args:
            angle_rad (_type_): _description_
            threshold (float, optional): direction error . Defaults to 0.5 (28.6degree)

        Returns:
            int : closest direction index
        """
        angles = self.normal_vector_angles()

        for i in range(4):
            drad = rad_distance(angles[i], angle_rad)
            if drad < threshold:
                return i
        # if no closest point 
        return None

    def get_equally_divided_coords(self,division_num,w,l):
        """divide one side

        Args:
            division_num (_type_): _description_
            w (_type_): width
            l (_type_): length

        
        division_num
        ---o-----o-----o-----o---- 
                                    ^
                                    |
                                    |
                                    |
                                    | length/2
                   center           |
                                    |
                     x              v
        <---------------------------->
                    width

        Returns:
            _type_: coordinates for divided points
        """
        divided_coords = []
        y = l/2
        x0 = - w/2
        for i in range(division_num):
            dx = (2*i + 1) / division_num/ 2 * w
            divided_coords.append([x0 + dx, y])
        return np.array(divided_coords).reshape(-1,2)

    def divide_coords(self, div_num, indx):
        """Return divided coordinate for estimating
        Note: 
            Assume lidar estimation is clock wise

        Args:
            div_num (_type_): number of measurement point in one side
            indx (_type_): side index 0 is front, 1 is right, 2 is back, 3 is left 

        Returns:
            coords(np.array): Nx2 row vector [[x1,y1].T,[x2, y2].T ...].T
        """
        R90 = rot_mat_2d(np.pi/2)
        R180 = rot_mat_2d(np.pi)
        center = np.array(self.center).reshape(-1,2) 
        
        if indx == 0:
            coords_ = self.get_equally_divided_coords(div_num,self.width,self.length)
            coords = center + coords_
        elif indx == 1:
            coords_ = self.get_equally_divided_coords(div_num,self.length,self.width)
            coords = center + np.transpose(R90.T @ coords_.T)
        elif indx == 2:
            coords_ = self.get_equally_divided_coords(div_num,self.width,self.length)
            coords = center + np.transpose(R180 @ coords_.T)
        elif indx == 3:
            coords_ = self.get_equally_divided_coords(div_num,self.length,self.width)
            coords = center + np.transpose(R90 @ coords_.T)
        else:
            logging.error()
            coords =  None
        return coords
    


# Vehicle model 1
class BicycleMotionModel():
    """state number should be 7 
        [x, y, v, psi , theta, l, w]
    """
    def __init__(self, angle_threshold=1e-9) -> None:
        self.cscv_boundary_angle = angle_threshold
        self.v_threshold = angle_threshold # used to avoid zero division in turning radius calc

    def predict(self,x, dt):
        theta = x[4]
        v = x[3]
        if theta < self.cscv_boundary_angle or v < self.v_threshold:
            x_ = self.predict_cv_model(x,dt)
        else:
            x_ = self.predict_cs_model(x,dt)
        return x_

    def predict_cs_model(self, x, dt):
        v = x[2]
        psi = x[3]
        theta = x[4]
        l = x[5]
        beta = dt * v/ l * np.tan(theta)
        TurnR = dt * v/ beta
        x_ = np.copy(x)

        x_[0] += TurnR * np.sin(psi+beta) - TurnR * np.sin(psi)
        x_[1] += TurnR * np.cos(psi+beta) +  TurnR * np.cos(psi)
        x_[3] += beta
        return x_

    def predict_cv_model(self, x, dt):
        v = x[2]
        psi = x[5]
        Dist = dt * v
        x_ = np.copy(x)

        x_[0] += Dist * np.cos(psi)
        x_[1] += Dist * np.sin(psi)
        x_[3] *= np.exp(-0.5)

        return x_

    def predict_noise_covariance(self, q_acc, q_yawrate, q_shape, dt):
        qp = dt*dt/2 * q_acc
        qv = dt * q_acc
        Q = np.diag([qp, qp, qv, q_yawrate*dt, q_yawrate,q_shape, q_shape])
        return Q
    

# Vehicle model 2
class ConstantVelocityModel():
    """state number should be 7 
    [x, y, vx, vy , phi, l, w]
    """
    def __init__(self) -> None:
        pass

    def predict(self, x, dt):
        A = np.diag([1]*7)
        A[0,2] = dt
        A[1,3] = dt

        return A @ x

    def predict_noise_covariance(self, q_acc, q_angle, q_shape, dt):
        qp = dt*dt/2 * q_acc
        qv = dt * q_acc
        Q = np.diag([qp, qp, qv, qv, q_angle, q_shape, q_shape])
        return Q