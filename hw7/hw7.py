#!/usr/bin/env python
from IPython.core.debugger import set_trace
from my_cv.visual_odometry import VO
from myholodeck.holodeck_plotter import OdometryPlotter
from myholodeck.uav_sim import UAVSim
import cv2
import numpy as np
import math
import transforms3d


urban_world = 'UrbanCity'
forest_world = 'EuropeanForest'
redwood_world = 'RedwoodForest'


camera_param_file = '/home/skyler/school/ecen631/camera_calibration/src/my_camera_calibration/param/webcam_intrinsic_parameters.yaml'

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
        plotter = OdometryPlotter(plotting_freq=15)
        uav_sim = UAVSim(forest_world)
        uav_sim.init_teleop()
        dt = 1.0/30.0
    else:
        uav_sim = UAVSim(forest_world)
        uav_sim.init_teleop()
        uav_sim.velocity_teleop = True
        dt = uav_sim.dt

        visual_odom = VO(process_nth_frame=1)
        plotter = OdometryPlotter(plotting_freq=1)



    while True:
        if using_webcam:
            # Process webcam frame
            ret, frame = cap.read()
            if ret == True:
                uav_sim.step_sim() # Just for plotter and to get time
                cv2.imshow("Webcam", frame)
                # cv2.imshow("Prev Frame", frame_prev)
                body_vel = np.array([1., 0., 0.])
                omega = np.array([0.0, 0.0, 0.1]) # Just so it estimates rotations
                Rhat, phat = visual_odom.estimate_odometry(frame, body_vel, omega, dt)
                euler = np.array(transforms3d.euler.mat2euler(Rhat, 'rxyz'))
                # Make appropriate changes for weird Holodeck frames
                xyz = np.copy(phat)
                xyz[2] *= -1.0
                euler[2] *= -1.0
                plotter.update_sim_data(uav_sim, xyz, euler)
        else:
            # Run holodeck
            uav_sim.step_sim()
            cam = uav_sim.get_camera()
            body_vel = uav_sim.get_body_velocity()
            omega = uav_sim.get_gyro()
            R = uav_sim.get_orientation()
            # Rhat, phat = visual_odom.estimate_odometry(cam, body_vel, omega, dt, R_truth=R) # Use true R
            Rhat, phat = visual_odom.estimate_odometry(cam, body_vel, omega, dt, show_features=True)
            euler = np.array(transforms3d.euler.mat2euler(Rhat, 'rxyz'))
            # Make appropriate changes for weird Holodeck frames
            xyz = np.copy(phat)
            xyz[2] *= -1.0
            euler[2] *= -1.0
            plotter.update_sim_data(uav_sim, xyz, euler)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            frame = np.copy(cam)
            # Process frame to get R,T difference
            visual_odom.compute_RT(frame)
            # Manually rotate to test rotation
            theta = 10
            R = cv2.getRotationMatrix2D((0,0), theta, 1.0)
            rows, cols, ch = frame.shape
            rot_frame = cv2.warpAffine(frame, R, (cols, rows))

            R, T = visual_odom.compute_RT(rot_frame)
            euler = transforms3d.euler.mat2euler(R, 'rxyz')
            print("T={0}, euler={1}".format(T, np.degrees(euler)))
            cv2.imshow("frame", frame)
            cv2.imshow("rotated frame", rot_frame)

        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visual_odometry_hw()
    print("Finished")
