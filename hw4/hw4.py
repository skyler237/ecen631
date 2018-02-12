import sys
rm_python2 = []
for p in sys.path:
    if p.find('python2') != -1:
        rm_python2.append(p)
for p in rm_python2:
    sys.path.remove(p)
sys.path.insert(0, '/home/skyler/school/ecen631/myholodeck')

import cv2
import numpy as np
import math
import time

from uav_sim import UAVSim
from my_cv import OpticalFlow
from optical_control import OpticalControl

urban_world = 'UrbanCity'
forest_world = 'EuropeanForest'
redwood_world = 'RedwoodForest'

def holodeck_sim():
    uav_sim = UAVSim(urban_world)
    uav_sim.init_teleop()
    uav_sim.init_plots(plotting_freq=15)
    uav_sim.velocity_teleop = True

    op_flow = OpticalFlow()
    control = OpticalControl()

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
        vy_c = control.follow_corridor(cam, display=True, ang_vel=ang_vel)
        yaw_c = uav_sim.yaw_c
        alt_c = 2.0

        uav_sim.command_velocity(vx_c, vy_c, yaw_c, alt_c)

        # cv2.imshow('Holodeck', cam)
        # cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    holodeck_sim()
    print("Finished")
