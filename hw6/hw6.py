#!/usr/bin/env python
from IPython.core.debugger import set_trace
from my_cv.visual_odometry import VO
from myholodeck.holodeck_plotter import CommandsPlotter
from myholodeck.uav_sim import UAVSim
import cv2
import numpy as np
import math




urban_world = 'UrbanCity'
forest_world = 'EuropeanForest'
redwood_world = 'RedwoodForest'


camera_param_file = '/home/skyler/school/ecen631/camera_calibration/src/my_camera_calibration/param/webcam_intrinsic_parameters.yaml'

def onClick(event, x, y, flags, param):
    global frame, frame_prev, visual_odom
    if event == cv2.EVENT_LBUTTONDOWN:
        # Process frame to get R,T difference
        visual_odom.compute_RT(frame)
        frame_prev = frame
        test_rotate = True
        if test_rotate:
            # Manually rotate to test rotation
            theta = 10
            R = cv2.getRotationMatrix2D((x,y), theta, 1.0)
            rows, cols, ch = frame.shape
            rot_frame = cv2.warpAffine(frame, R, (cols, rows))
            visual_odom.compute_RT(rot_frame)
            # Update previous frame
            frame_prev = rot_frame

def nothing():
    pass

def visual_odometry_hw():
    using_webcam = False

    # System setup
    if using_webcam:
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Webcam")
        cv2.setMouseCallback("Webcam", onClick)
        # Get region of interest
        ret, frame_prev = cap.read()
        visual_odom = VO(camera_param_file)
    else:
        uav_sim = UAVSim(redwood_world)
        uav_sim.init_teleop()
        uav_sim.velocity_teleop = True

        visual_odom = VO()
        plotter = CommandsPlotter(plotting_freq=10)

        dt = 1.0/30.0


    while True:
        if using_webcam:
            # Process webcam frame
            ret, frame = cap.read()
            if ret == True:
                cv2.imshow("Webcam", frame)
                cv2.imshow("Prev Frame", frame_prev)
        else:
            # Run holodeck
            uav_sim.step_sim()
            cam = uav_sim.get_camera()
            plotter.update_sim_data(uav_sim)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visual_odometry_hw()
    print("Finished")
