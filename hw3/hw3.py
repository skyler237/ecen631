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
forest_world = 'ConiferForest'

edge_min = 150
edge_max = 200

def holodeck_sim():
    uav_sim = UAVSim(urban_world)
    uav_sim.init_teleop()
    uav_sim.init_plots()

    while True:
        uav_sim.step_sim()
        cam = uav_sim.get_camera()
        gray = cv2.cvtColor(cam, cv2.COLOR_RGBA2GRAY)
        edge = cv2.Canny(cam, edge_min, edge_max)
        bgr = cv2.cvtColor(cam, cv2.COLOR_RGBA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # top = np.hstack((cam,gray))
        # bot = np.hstack((edge,hsv))
        # img = np.vstack((top,bot))
        cv2.imshow('Gray', gray)
        cv2.imshow('Edge', edge)
        cv2.imshow('HSV', hsv)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    holodeck_sim()
    print("Finished")
