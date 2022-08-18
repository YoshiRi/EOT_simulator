""" 

Vehicle and LIDAR Simulator

partly migrated from following projects
https://github.com/AtsushiSakai/PythonRobotics
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import random

from utils import rot_mat_2d
from tracker.LshapeFitting import LShapeFitting


LIDAR_NOISE_SIGMA = 0.01
show_animation = True

class PerceptionSimulator():
    def __init__(self, dt=0.1, sim_time=30, angle_resolution_deg = 3.0,show_animation=True):
        """Simulator setting
        """
        self.dt = dt
        self.sim_time = sim_time
        self.angle_resolution = np.deg2rad(angle_resolution_deg) # LIDAR sensor resolution
        self.show_animation = show_animation 
        self.vehicle_sets = []        # managing vehicle simulators

    
    def append_vehicle(self,vehicle_simulator,velocity_set=None):
        """ append vehicle
        """
        if velocity_set is None:
            velocity_set = [0.1, 0.0]
        
        sim_set = {}
        sim_set["vehicle"] = vehicle_simulator
        sim_set["velocity"] = velocity_set
        self.vehicle_sets.append(sim_set)


    def run(self,detector):
        """
        Constant Velocity: Set
        """
        time = 0.0
        lidar_sim = LidarSimulator(LIDAR_NOISE_SIGMA)

        while time <= self.sim_time:
            time += self.dt

            for vs in self.vehicle_sets:
                vs["vehicle"].update(self.dt,*vs["velocity"])

            # observation
            vehicles = [vs["vehicle"] for vs in self.vehicle_sets]
            ox, oy = lidar_sim.get_observation_points(vehicles, self.angle_resolution)

            # fitting
            shapes, id_sets = detector.fitting(ox, oy)

            if self.show_animation: 
                self.visualize(shapes, id_sets, ox, oy)
        print("Done")


    
    def visualize(self, shapes, id_sets,  ox, oy):
        """
        Plot simulation
        shapes: detected shape object
        id_sets: ??
        ox, oy: lidar data
        """                
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.axis("equal")
        plt.plot(0.0, 0.0, "*r")

        # draw vehicle
        for vs in self.vehicle_sets:
            vs["vehicle"].plot()

    
        # Plot range observation
        for ids in id_sets:
            x = [ox[i] for i in range(len(ox)) if i in ids]
            y = [oy[i] for i in range(len(ox)) if i in ids]

            for (ix, iy) in zip(x, y):
                plt.plot([0.0, ix], [0.0, iy], "-og")

            plt.plot([ox[i] for i in range(len(ox)) if i in ids],
                    [oy[i] for i in range(len(ox)) if i in ids],
                    "o")
        
        for shape in shapes:
            shape.plot()

        plt.pause(0.02)


class VehicleSimulator:
    """define vehicle model
    """

    def __init__(self, i_x, i_y, i_yaw, i_v, max_v, w, L):
        self.x = i_x
        self.y = i_y
        self.yaw = i_yaw
        self.v = i_v
        self.max_v = max_v
        self.W = w
        self.L = L
        self._calc_vehicle_contour()

    def update(self, dt, a, omega):
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += omega * dt
        self.v += a * dt
        if self.v >= self.max_v:
            self.v = self.max_v

    def plot(self):
        plt.plot(self.x, self.y, ".b")

        # convert global coordinate
        gx, gy = self.calc_global_contour()
        plt.plot(gx, gy, "--b")

    def calc_global_contour(self):
        gxy = np.stack([self.vc_x, self.vc_y]).T @ rot_mat_2d(self.yaw)
        gx = gxy[:, 0] + self.x
        gy = gxy[:, 1] + self.y

        return gx, gy

    def _calc_vehicle_contour(self):

        self.vc_x = []
        self.vc_y = []

        self.vc_x.append(self.L / 2.0)
        self.vc_y.append(self.W / 2.0)

        self.vc_x.append(self.L / 2.0)
        self.vc_y.append(-self.W / 2.0)

        self.vc_x.append(-self.L / 2.0)
        self.vc_y.append(-self.W / 2.0)

        self.vc_x.append(-self.L / 2.0)
        self.vc_y.append(self.W / 2.0)

        self.vc_x.append(self.L / 2.0)
        self.vc_y.append(self.W / 2.0)

        self.vc_x, self.vc_y = self._interpolate(self.vc_x, self.vc_y)

    @staticmethod
    def _interpolate(x, y):
        rx, ry = [], []
        d_theta = 0.05
        for i in range(len(x) - 1):
            rx.extend([(1.0 - theta) * x[i] + theta * x[i + 1]
                       for theta in np.arange(0.0, 1.0, d_theta)])
            ry.extend([(1.0 - theta) * y[i] + theta * y[i + 1]
                       for theta in np.arange(0.0, 1.0, d_theta)])

        rx.extend([(1.0 - theta) * x[len(x) - 1] + theta * x[1]
                   for theta in np.arange(0.0, 1.0, d_theta)])
        ry.extend([(1.0 - theta) * y[len(y) - 1] + theta * y[1]
                   for theta in np.arange(0.0, 1.0, d_theta)])

        return rx, ry


class LidarSimulator:

    def __init__(self,range_noise=0.01):
        """Lidar Simulator with Uniform distributed noise observation
        Noise: Uniform distribution, multiplicative noise
        Args:
            range_noise (float, optional): multiplicative noise width. Defaults to 0.01.
        """
        self.range_noise = range_noise

    def get_observation_points(self, v_list, angle_resolution):
        x, y, angle, r = [], [], [], []

        # store all points
        for v in v_list:

            gx, gy = v.calc_global_contour()

            for vx, vy in zip(gx, gy):
                v_angle = math.atan2(vy, vx)
                vr = np.hypot(vx, vy) * random.uniform(1.0 - self.range_noise,
                                                       1.0 + self.range_noise)

                x.append(vx)
                y.append(vy)
                angle.append(v_angle)
                r.append(vr)

        # ray casting filter
        rx, ry = self.ray_casting_filter(angle, r, angle_resolution)

        return rx, ry

    @staticmethod
    def ray_casting_filter(theta_l, range_l, angle_resolution):
        rx, ry = [], []
        range_db = [float("inf") for _ in range(
            int(np.floor((np.pi * 2.0) / angle_resolution)) + 1)]

        for i in range(len(theta_l)):
            angle_id = int(round(theta_l[i] / angle_resolution))

            if range_db[angle_id] > range_l[i]:
                range_db[angle_id] = range_l[i]

        for i in range(len(range_db)):
            t = i * angle_resolution
            if range_db[i] != float("inf"):
                rx.append(range_db[i] * np.cos(t))
                ry.append(range_db[i] * np.sin(t))

        return rx, ry



def senario1():

    sim = PerceptionSimulator(dt=0.2)

    v1 = VehicleSimulator(-10.0, 0.0, np.deg2rad(90.0),
                          0.0, 50.0 / 3.6, 3.0, 5.0)

    sim.append_vehicle(v1)


    l_shape_fitting = LShapeFitting()

    sim.run(l_shape_fitting)
    print("Done")


if __name__ == '__main__':
    senario1()
