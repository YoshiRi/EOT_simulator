"""
Rectangle Shape Tracker with Pure EKF


http://liu.diva-portal.org/smash/get/diva2:434601/FULLTEXT02.pdf


Yoshi Ri
2022/08/18
"""

from ast import Constant
from selectors import EpollSelector
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import numpy as np
from utils import RectangleData
from EKF import ExtendedKalmanFilter
from RectangleTracker import *

from simulator import PerceptionSimulator, VehicleSimulator


class EKFRectangleTracker(ExtendedKalmanFilter):
    def __init__(self) -> None:
        super().__init__()
        self.x = np.array([0.0]*7).reshape(-1,1) 
        self.P = np.diag([1e4,1e4,1e9,1e2,1e9,1e2,1e2])
        self.set_shape(1,1) # shape should larger than 0
        self.sensor_noise =1e-1
        self.dt = 1e-1
        self.log = []

    def set_sampling_time(self,dt):
        self.dt = dt

    def set_pos(self,pose):
        self.x[0] = pose[0]
        self.x[1] = pose[1]

    def set_vel(self,vel):
        self.x[2] = vel[0]
        self.x[3] = vel[1]

    def set_vel_bicycle(self,vel):
        self.x[2] = vel

    def set_orientation_bicycle(self,orientation, wheel_angle = None):
        self.x[3] = orientation
        self.x[4] = wheel_angle if not wheel_angle is None else 0
    
    def set_orientation(self, orientation):
        self.x[4] = orientation

    def set_shape(self,length,width):
        self.x[5] = length
        self.x[6] = width

    def set_model(self,motion_model=ConstantVelocityModel):
        self.motion_model = motion_model()

    def measurement_process(self, z, dt):

        Qnoise = self.motion_model.predict_noise_covariance(1e2,1e-1,1e-9,dt) #pos(acc) cov, rot cov, shape cov
        x_, P_ = self.predict_nonlinear(self.motion_model.predict, Qnoise, dt=dt)
        
        # reshape measurement as row vector
        measurements = np.array(z).reshape(-1,1)

        # show estimation
        import matplotlib.pyplot as plt
        plt.plot(measurements.reshape(-1,2)[:,0], measurements.reshape(-1,2)[:,1],'x')

        #estimated_measurements = get_estimated_rectangular_points(self.x, measurements).reshape(-1,2)
        estimated_measurements = calc_rectangle_counter(x_, measurements).reshape(-1,2)
        plt.plot(estimated_measurements[:,0], estimated_measurements[:,1],'ko')
        #plt.show()

        if len(z) > 0:
            Rnoise = np.diag([self.sensor_noise]*len(z)*2)
            #self.update_nonlinear(x_, P_, get_estimated_rectangular_points,measurements, Rnoise, measurements=measurements)
            self.update_nonlinear(x_, P_, calc_rectangle_counter, measurements, Rnoise,  measurements=measurements)
            
        else:
            self.x = x_
            self.P = P_

    
    def fitting(self,ox,oy):
        """This function is needed to run simulation 

        Args:
            ox (_type_): x coordinate of lidar observation
            oy (_type_): y coordinate of lidar observation
            dt (_type_): sampling tme
            tau (int, optional): tuning parameter for shape prediction agility. larger tau make shape estimation faster. Defaults to 2.
            process_noise (int, optional): Random acceleration for object [m/s/s].  Defaults to 10.

        Returns:
            shape_obj, associated_ids 
        """
        dt = self.dt
        assert len(ox)==len(oy), "Measurement must have same number of x and y axis value."
        z = [[x,y] for x,y in zip(ox,oy)]

        self.measurement_process(z,dt)
        shape = RectangleData()
        shape.center = self.x[0:2]
        shape.length = self.x[5]
        shape.width = self.x[6]
        shape.orientation = self.x[4]
        ids = np.arange(0, len(ox))

        self.log.append([self.x, self.P])
        return [shape], [ids]


    def __del__(self):
        import matplotlib.pyplot as plt
        plt.figure(2)
        x = [log[0][0] for log in self.log]
        t = np.arange(len(x))
        y = [log[0][1] for log in self.log]
        v = [log[0][2] for log in self.log]
        cta = [log[0][3] for log in self.log]
        beta = [log[0][4] for log in self.log]
        l = [log[0][5] for log in self.log]
        w = [log[0][6] for log in self.log]
        plt.plot(t,x,t,y)
        plt.legend(["x","y"])
        plt.figure(3)
        plt.plot(t,v,t,cta,t,beta)
        plt.legend(["vx","vy","orientation","handle angle"])
        plt.figure(4)
        plt.plot(t,l,t,w)
        plt.legend(["len","wid"])
        plt.show()




def calc_rectangle_counter(x, measurements):
    obj = calcRectangleCounter()
    obj.init_from_constant_velocity_model_state(x)
    z_est = obj.calc_measurement_points(measurements)
    return z_est 

class calcRectangleCounter(RectangleShapePrediction):
    def __init__(self) -> None:
        pass

    def init_with_param(self,center,orientation,l,w):
        self.center = np.array(center).reshape(2,1).astype(float)
        self.orientation = orientation
        self.l = float(l)
        self.w = float(w)

    def init_from_constant_velocity_model_state(self,x_state):
        center = x_state[0:2]
        psi = x_state[4]
        self.init_with_param(center,psi,x_state[5],x_state[6])        

    def calcS(self, idx, n, Nz):
        SW = np.array([0,1,1,0]).reshape(2,2)
        S1 = np.diag([0.5, -(2.*n+1-Nz)/Nz/2.]).reshape(2,2)
        S2 = rot_mat_2d(-np.pi/2) @ S1 @ SW
        S3 = rot_mat_2d(np.pi) @ S1
        S4 = rot_mat_2d(np.pi/2) @ S1 @ SW
        S = [S1,S2,S3,S4]
        return S[idx]

    def dRmatrix(self,rad):
        """2d rotation derivative"""
        dR = np.array([-np.sin(rad), -np.cos(rad), np.cos(rad), -np.sin(rad)]).reshape(2,2)
        return dR

    def calc_JacobH_part(self, parted_z):
        """Calc jacobian from measurements set z

        Args:
            parted_z (_type_): measurement array

        Returns:
            _type_: jacobian
        """
        pz = np.array(parted_z).reshape(-1,2)
        nvec_rad = measurements_normalvec_angle(pz)
        idx = self.find_closest_angle(nvec_rad)

        R = rot_mat_2d(self.orientation)
        dR = self.dRmatrix(self.orientation)
        lwvec = np.array([self.l, self.w]).reshape(2,1)

        J_H = np.array([])
        Nz = pz.shape[0]
        for i in range(Nz):
            S = self.calcS(idx, i, Nz)
            J_Hn = np.hstack([np.eye(2), dR @ S @ lwvec, np.zeros((2,1)), R @ S ])

            J_H = np.vstack([J_H, J_Hn]) if J_H.shape[0] else J_Hn
        return J_H

    def calc_JacobH(self, measurement):
        """Calc Jacobian from measurement

        Args:
            measurement (_type_): _description_

        Returns:
            _type_: _description_
        """
        Z = np.array(measurement).reshape(-1,2)
        side_num = estimate_number_of_sides(Z)
        J_H = np.array([])
        
        if side_num == 1:
            J_H = self.calc_JacobH_part(Z)
        elif side_num == 2:
            corner_index, parted_measurements = find_corner_index(Z)
            for pm in parted_measurements:
                J_Hn = self.calc_JacobH_part(pm)
                J_H  = np.vstack([J_H, J_Hn]) if J_H.size else J_Hn
        else:
            print("side num must be 1 or 2! current value is: ", side_num)
            sys.exit(-1)

        return J_H

    def calc_measurement_points_part(self, parted_z, idx):
        """Calc jacobian from measurements set z

        Args:
            parted_z (_type_): measurement array

        Returns:
            _type_: jacobian
        """
        pz = np.array(parted_z).reshape(-1,2)

        R = rot_mat_2d(self.orientation)
        lwvec = np.array([self.l, self.w]).reshape(2,1)

        Z_hat = np.array([])
        Nz = pz.shape[0]
        for i in range(Nz):
            S = self.calcS(idx, i, Nz)
            z_est = R @ S @ lwvec + self.center

            Z_hat = np.vstack([Z_hat, z_est]) if Z_hat.size else z_est
        return Z_hat


    def calc_measurement_points(self, measurement):
        Z = np.array(measurement).reshape(-1,2)
        side_num = estimate_number_of_sides(Z)
        Z_hat = np.array([])
        
        if side_num == 1:
            idx = self.guess_measurements_side(Z)
            Z_hat = self.calc_measurement_points_part(Z,idx)
        elif side_num == 2:
            corner_index, parted_measurements = find_corner_index(Z)
            idxs = self.guess_measurements_sides(parted_measurements, corner_index)
            for pm,idx in zip(parted_measurements, idxs):
                #idx = self.guess_measurements_side(pm)
                Z_Hn = self.calc_measurement_points_part(pm,idx)
                Z_hat  = np.vstack([Z_hat, Z_Hn]) if Z_hat.size else Z_Hn
        else:
            print("side num must be 1 or 2! current value is: ", side_num)
            sys.exit(-1)

        # When N = 3 (corner is duplicated)
        if Z_hat.shape[0] == Z.shape[0]+1:
            Z_hat = np.delete(Z_hat,corner_index, axis=0) # remove duplicated corner

        return Z_hat

    def guess_measurements_side(self, measurement):
        Z = np.array(measurement).reshape(-1,2)
        nvec_rad = measurements_normalvec_angle(Z)
        idx = self.find_closest_angle(nvec_rad)
        return idx

    def guess_measurements_sides(self, measurements, corner_index):
        Z1 = np.array(measurements[0]).reshape(-1,2)
        Z2 = np.array(measurements[1]).reshape(-1,2)
        Z = np.vstack([Z1, Z2])
        
        nvec_rad = measurements_normalvec_angle([Z1[0], Z[corner_index]])
        idx1 = self.find_closest_angle(nvec_rad)
        nvec_rad = measurements_normalvec_angle([Z[corner_index], Z2[-1]])
        idx2 = self.find_closest_angle(nvec_rad)

        return [idx1, idx2]




def senario1():
    sim = PerceptionSimulator(dt=0.1)
    v1 = VehicleSimulator(-10.0, 10.0, np.deg2rad(90.0),
                          0.0, 50.0 / 3.6, 3.0, 5.0)
    vref = [0.1, 0]
    sim.append_vehicle(v1,vref)

    tracker = EKFRectangleTracker()
    tracker.set_model()
    tracker.set_shape(5,2)
    tracker.set_pos([-10,10])
    tracker.set_orientation(np.pi/2)

    sim.run(tracker)
    print("Done")

def senario2():
    sim = PerceptionSimulator(dt=0.1)
    v1 = VehicleSimulator(-10.0, 30.0, np.deg2rad(90.0),
                          0.0, 50.0 / 3.6, 3.0, 5.0)
    vref = [-0.1, 0]
    sim.append_vehicle(v1,vref)

    tracker = EKFRectangleTracker()
    tracker.set_model()
    tracker.set_shape(5,3)
    tracker.set_pos([-10,30])
    tracker.set_orientation(np.pi/2)

    sim.run(tracker)
    print("Done")


def senario3():
    sim = PerceptionSimulator(dt=0.1)
    v1 = VehicleSimulator(-10.0, 0.0, np.deg2rad(90.0),
                          0.0, 50.0 / 3.6, 3.0, 5.0)
    vref = [0.1, 0.02]
    sim.append_vehicle(v1,vref)

    tracker = EKFRectangleTracker()
    tracker.set_model()
    tracker.set_shape(5,3)
    tracker.set_pos([-10,0])
    tracker.set_orientation(np.pi/2)

    sim.run(tracker)
    print("Done")

if __name__=="__main__":
    #senario1()
    #senario2()
    senario3()