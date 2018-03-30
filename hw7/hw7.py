#!/usr/bin/env python
from IPython.core.debugger import set_trace
from my_cv.visual_odometry import VO
from my_cv.reconstruct_3d import Reconstruct3D
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
dataset_param_file = '/home/skyler/school/ecen631/hw7/templeRing/camera_params.yaml'
dataset_img_file = lambda i: '/home/skyler/school/ecen631/hw7/templeRing/templeR000{0}.png'.format(i)

def visual_odometry_hw():
    using_webcam = True
    using_dataset = False

    # System setup
    if using_webcam:
        reconstructor = Reconstruct3D(camera_param_file)
    elif using_dataset:
        initial_img_index = 6
        final_img_index = 12
        reconstructor = Reconstruct3D(dataset_param_file)
    else:
        uav_sim = UAVSim(urban_world)
        uav_sim.init_teleop()
        uav_sim.velocity_teleop = True
        dt = uav_sim.dt

        frame_cnt = 0
        process_nth_frame = 1
        reconstructor = Reconstruct3D()



    while True:
        if using_webcam:
            # Process webcam frame
            ret, frame = cap.read()
            if ret == True:
                cv2.imshow("Webcam", frame)
                body_vel = np.array([1., 0., 0.])
                reconstructor.get_3d_points2(cam, body_vel=body_vel)
        elif using_dataset:
            # Cycle through images
            for i in range(initial_img_index, final_img_index+1):
                img = cv2.imread(dataset_img_file(i))
                reconstructor.get_3d_points(img)

            # Exit the loop
            break
        else:
            # Run holodeck
            uav_sim.step_sim()
            cam = uav_sim.get_camera()
            body_vel = uav_sim.get_body_velocity()
            omega = uav_sim.get_gyro()
            R = uav_sim.get_orientation()
            T = uav_sim.get_position()
            T[2] *= -1
            # reconstructor.get_3d_points2(cam, body_vel=body_vel, R_truth=R, T_truth=T)
            reconstructor.get_3d_points2(cam, body_vel=body_vel, R_truth=R, T_truth=T, use_truth=True)

            # reconstructor.get_3d_points2(cam, body_vel=body_vel)

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
