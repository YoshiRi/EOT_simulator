"""
Rectangle Shape Tracker with Pure EKF


http://liu.diva-portal.org/smash/get/diva2:434601/FULLTEXT02.pdf


Yoshi Ri
2022/08/18
"""

from ast import Constant
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
        estimated_measurements = get_estimated_rectangular_points(self.x, measurements).reshape(-1,2)
        import matplotlib.pyplot as plt
        plt.plot(estimated_measurements[:,0], estimated_measurements[:,1],'o')
        #plt.show()

        if len(z) > 0:
            Rnoise = np.diag([self.sensor_noise]*len(z)*2)
            self.update_nonlinear(x_, P_, get_estimated_rectangular_points,measurements, Rnoise, measurements=measurements)
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
    tracker.set_shape(6,4)
    tracker.set_pos([-10,30])
    tracker.set_orientation(np.pi/2)

    sim.run(tracker)
    print("Done")


if __name__=="__main__":
    #senario1()
    senario2()