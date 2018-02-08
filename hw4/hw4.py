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

urban_world = 'UrbanCity'
forest_world = 'EuropeanForest'
redwood_world = 'RedwoodForest'

def holodeck_sim():
    uav_sim = UAVSim(urban_world)
    uav_sim.init_teleop()
    uav_sim.init_plots(plotting_freq=1)
    uav_sim.command_velocity = True

    while True:
        uav_sim.step_sim()
        cam = uav_sim.get_camera()

        cv2.imshow('Holodeck', cam)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    holodeck_sim()
    print("Finished")
