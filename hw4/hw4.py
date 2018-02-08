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

urban_world = 'UrbanCity'
forest_world = 'EuropeanForest'
redwood_world = 'RedwoodForest'

def holodeck_sim():
    uav_sim = UAVSim(urban_world)
    uav_sim.init_teleop()
    uav_sim.init_plots(plotting_freq=5)
    uav_sim.command_velocity = True

    op_flow = OpticalFlow()

    for i in range(0,5):
        uav_sim.step_sim()

    while True:
        uav_sim.step_sim()
        cam = uav_sim.get_camera()

        # print(np.shape(cam))
        #
        # bgr = cv2.cvtColor(cam, cv2.COLOR_RGBA2BGR)
        # gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Implement optical flow
        op_flow.compute_optical_flow(cam)
        op_flow.display_image()

        # cv2.imshow('Holodeck', cam)
        # cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    holodeck_sim()
    print("Finished")
