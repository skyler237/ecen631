from myholodeck.uav_sim import UAVSim
from myholodeck.optical_control import OpticalControl
from my_cv.optical_flow import OpticalFlow

import cv2
import numpy as np
import math
import time


urban_world = 'UrbanCity'
forest_world = 'EuropeanForest'
redwood_world = 'RedwoodForest'

def holodeck_sim():
    uav_sim = UAVSim(redwood_world)
    uav_sim.init_teleop()
    # uav_sim.init_plots(plotting_freq=10)
    uav_sim.velocity_teleop = True

    op_flow = OpticalFlow()
    control = OpticalControl()
    dt = 1.0/30.0

    for i in range(0,5):
        uav_sim.step_sim()

    while True:
        uav_sim.step_sim()
        cam = uav_sim.get_camera()

        # print(np.shape(cam))
        #
        # bgr = cv2.cvtColor(cam, cv2.COLOR_RGBA2BGR)
        # gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        ang_vel = uav_sim.get_imu()[3:6]
        vx_c = 3.0
        # vy_c = 0.0
        vy_c = control.follow_corridor(cam, display=False, ang_vel=ang_vel)
        # yaw_c = uav_sim.yaw_c
        yaw_c = control.avoid_obstacles(cam, uav_sim.yaw_c, dt, display=False, ang_vel=ang_vel)
        # alt_c = 2.5
        alt_c = control.altitude_hold(cam, uav_sim.alt_c, dt, display=False, ang_vel=ang_vel)

        uav_sim.command_velocity(vx_c, vy_c, yaw_c, alt_c)

        # cv2.imshow('Holodeck', cam)
        # cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    holodeck_sim()
    print("Finished")
