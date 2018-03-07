from IPython.core.debugger import set_trace
from my_cv.cv_utils import *
from my_cv.feature_tracker import KLTTracker
import numpy as np
import transforms3d
import cv2
import math
import yaml


class VO:
    def __init__(self, cam_param_file=None):
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
        self.ransac_thresh = 1.0
        self.ransac_prob = 0.999

        # Other params
        self.min_inliers = 50

        # Initialize KLT tracker
        self.klt = KLTTracker(max_features=4000, dt=1.0/30.0)
        self.klt.default_img_size = self.img_size
        self.klt.default_img_type = self.img_type

        # Initialize state
        self.phat = np.array([0., 0., 0.])  # estimated position
        self.Rhat = np.eye(3)               # estimated rotation (matrix)

        self.pose = self.construct_pose_matrix(self.Rhat, self.phat)

        self.R_cam2body = np.array([[0., 0., 1.],
                                    [1., 0., 0.],
                                    [0., -1., 0.]])
        # # FIXME: Just for testing
        # self.R_cam2body = np.eye(3)


    def compute_RT(self, frame):
        # Get feature matches
        # REVIEW: Are these flipped? why?
        features, prev_features = self.klt.get_feature_matches(frame, self.img_type)
        display_features(frame, features)

        E, mask = cv2.findEssentialMat(prev_features, features, self.K,
                                        method=cv2.RANSAC, threshold=self.ransac_thresh, prob=self.ransac_prob)
        inliers = np.sum(mask)
        if inliers <= self.min_inliers:
            print("Not enough inliers ({0} < {1})".format(inliers, self.min_inliers))
            return None, None
        ret, R, T, mask = cv2.recoverPose(E, prev_features, features, self.K, mask=mask)

        euler = np.degrees(transforms3d.euler.mat2euler(R, 'rxyz'))

        return R, T

    def estimate_odometry(self, frame, body_vel, dt, R_truth=None):
        # Get relative transformation
        R_cam, T_cam = self.compute_RT(frame)

        # Check if the pose recovery was successful
        if R_cam is not None:
            # Rotate both into body frame
            R_body = np.matmul(self.R_cam2body, R_cam)
            T_body = np.matmul(self.R_cam2body, T_cam)

            # Scale T using body_velocity
            T_scaled = T_body*np.linalg.norm(body_vel)*dt
            R_scaled = R_body
            # same_sign = np.equal(np.sign(T_scaled), np.sign(body_vel))
            # # Check to see if the algorithm returned the wrong sign on T
            # if np.sum(same_sign) < 3:
            #     T_scaled *= -1

            C_body = self.construct_pose_matrix(R_scaled, T_scaled)

            # Update pose
            self.pose = np.matmul(self.pose, C_body)

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
        # return self.Rhat, self.phat
        return self.Rhat, self.phat
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
