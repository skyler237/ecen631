from IPython.core.debugger import set_trace
from my_cv.cv_utils import *
from my_cv.feature_tracker import KLTTracker
import numpy as np
import transforms3d
import cv2
import math
import yaml


class VO:
    def __init__(self, cam_param_file=None, process_nth_frame=1):
        if cam_param_file is None:
            # Default camera parameters -- Holodeck
            self.img_size = (512,512,4)
            self.img_type = 'rgba'
            self.f = 512/2
            self.K = np.array([[self.f, 0.0,    512/2],
                               [0.0,    self.f, 512/2],
                               [0.0,    0.0,    1.0]])
        else:
            # Extract camera params
            cam_params = CamParams(cam_param_file)
            self.K = cam_params.K
            self.img_size = (cam_params.width,cam_params.height,3)
            self.img_type = 'bgr'
            self.f = (self.K[0][0] + self.K[1][1])/2.0 # Take the average of fx and fy

        # RANSAC params
        self.ransac_thresh = 0.1
        self.ransac_prob = 0.999

        # Other params
        self.min_inliers = 50
        self.omega_thresh = math.radians(0.2)
        self.feature_motion_threshold = 1.75 #px
        self.euler_threshold = math.radians(10.0)
        self.process_nth_frame = process_nth_frame

        # Initialize KLT tracker
        self.klt = KLTTracker(max_features=5000, min_features=2000, dt=1.0/30.0, img_size=self.img_size, img_type=self.img_type)
        self.klt.default_img_size = self.img_size
        self.klt.default_img_type = self.img_type

        # Initialize state
        self.phat = np.array([0., 0., 0.])  # estimated position
        self.Rhat = np.eye(3)               # estimated rotation (matrix)
        self.frame_cnt = 0

        self.pose = self.construct_pose_matrix(self.Rhat, self.phat)

        self.R_cam2body = np.array([[0.,  0.,  1.],
                                    [1.,  0.,  0.],
                                    [0.,  1.,  0.]])
        # # FIXME: Just for testing
        # self.R_cam2body = np.eye(3)


    def compute_RT(self, frame, show_features=False):
        # Get feature matches
        # REVIEW: Are these flipped? why?
        features, prev_features = self.klt.get_feature_matches(frame, self.img_type, motion_thresh=self.feature_motion_threshold)
        if show_features:
            display_features(frame, features)

        if len(features) < self.min_inliers:
            # print("Not enough inliers ({0} < {1})".format(len(features), self.min_inliers))
            return None, None

        E, mask = cv2.findEssentialMat(prev_features, features, self.K,
                                        method=cv2.RANSAC, threshold=self.ransac_thresh, prob=self.ransac_prob)
        inliers = np.sum(mask)
        if inliers < self.min_inliers:
            # print("Not enough inliers ({0} < {1})".format(inliers, self.min_inliers))
            return None, None
        ret, R, T, mask = cv2.recoverPose(E, prev_features, features, self.K, mask=mask)

        euler = np.degrees(transforms3d.euler.mat2euler(R, 'rxyz'))

        return R, T

    def estimate_odometry(self, frame, body_vel, omega, dt, R_truth=None, show_features=False):
        self.frame_cnt += 1
        if self.frame_cnt % self.process_nth_frame != 0:
            return self.Rhat, self.phat

        # Get relative transformation
        R_cam, T_cam = self.compute_RT(frame, show_features=show_features)

        # Check if the pose recovery was successful
        if R_cam is not None:
            # Rotate both into body frame
            R_body = np.dot(self.R_cam2body, R_cam).dot(self.R_cam2body.transpose())
            T_body = np.dot(self.R_cam2body, T_cam)

            # Check for spikes
            euler = np.array(transforms3d.euler.mat2euler(R_body, 'rxyz'))
            if np.max(np.abs(euler)) > self.euler_threshold:
                # Skip this iteration
                return self.Rhat, self.phat

            # Zero out axes that have too little rotation
            euler[np.abs(euler) < self.omega_thresh] = 0.0
            # euler[np.abs(omega) < self.omega_thresh] = 0.0
            R_body = transforms3d.euler.euler2mat(*euler)

            # Scale T using body_velocity
            extra_scale = 0.65
            T_scaled = T_body*np.linalg.norm(body_vel)*dt*extra_scale

            C_body = self.construct_pose_matrix(R_body, T_scaled)

            # Update pose
            self.pose = np.matmul(self.pose, C_body)
            print("T_cam = {0}".format(T_cam))
            print("T_body = {0}".format(T_body))
            print("T_scaled = {0}".format(T_scaled))


            # # FIXME: Just for testing!
            # self.pose = self.construct_pose_matrix(R_body, T_body)

            # FIXME: For testing translation
            if R_truth is not None:
                self.pose[0:3,0:3] = R_truth

            # # Rotate T into the world frame
            # if R_truth is not None:
            #     T_world = np.ravel(np.dot(R_truth.transpose(),T_scaled)) # Rotate translation into world frame
            # else:
            #     T_world = np.ravel(np.dot(self.Rhat.transpose(),T_scaled)) # Rotate translation into world frame
            #
            # # Update position estimate
            # self.phat += T_world
            #
            # # Update orientation estimate
            # self.Rhat = np.matmul(R_body, self.Rhat)

        self.Rhat, self.phat = self.decompose_pose_matrix(self.pose)
        print("phat = {0}".format(self.phat))
        # return self.Rhat, self.phat
        return self.Rhat, self.phat
        # return R_cam, T_cam
        # return R_body, T_scaled

    def construct_pose_matrix(self, R, T):
        # Make sure T is a column vector
        T = np.reshape(T,(3,1))
        C = np.vstack((np.hstack((R,T)),[0., 0., 0., 1.]))
        return C

    def decompose_pose_matrix(self, C):
        R = C[0:3,0:3]
        T = C[0:3,3]
        return R, T
