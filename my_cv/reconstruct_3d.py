from IPython.core.debugger import set_trace
from my_cv.cv_utils import *
from my_cv.buffer import *
from my_cv.feature_tracker import KLTTracker
import numpy as np
import transforms3d
import cv2
import math
import yaml
import pyqtgraph as pg
import pyqtgraph.opengl as gl

class Reconstruct3D:
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

        # Initialize frame buffer
        self.window_size = 5
        self.frame_buffer = FrameBuffer(self.window_size)

        # Initialize KLT tracker
        self.klt = KLTTracker(max_features=5000, min_features=2000, dt=1.0 / 30.0, img_size=self.img_size,
                              img_type=self.img_type)

        # RANSAC params
        self.ransac_thresh = 0.1
        self.ransac_prob = 0.999

        # Other params
        self.min_inliers = 50
        self.omega_thresh = math.radians(0.2)
        self.feature_motion_threshold = 1.00 #px
        self.euler_threshold = math.radians(10.0)

        # Initialize state buffers
        self.T_buffer = Buffer(self.window_size)
        T_init = np.zeros((3,1))  # estimated position
        self.T_buffer.push(T_init)

        self.R_buffer = Buffer(self.window_size)
        R_init = np.eye(3)               # estimated rotation (matrix)
        self.R_buffer.push(R_init)

        self.R_cam2body = np.array([[0.,  0.,  1.],
                                    [1.,  0.,  0.],
                                    [0.,  1.,  0.]])

        app = pg.QtGui.QApplication([])
        self.plot_window = gl.GLViewWidget()


    def get_3d_points(self, frame):
        self.frame_buffer.add_frame(frame)
        if self.frame_buffer.cnt() == 1:
            # Initialize the feature tracker
            self.klt.get_feature_matches(frame,self.img_type, self.feature_motion_threshold)
        else:
            # Get feature matches
            feature_matches = self.klt.get_feature_matches(frame, self.img_type,
                                                                   motion_thresh=self.feature_motion_threshold)

            # Compute rotation and translation between last two frames
            R_cam, T_cam = self.compute_RT(frame, feature_matches=feature_matches)

            # Get previous projection matrix
            R_prev = self.R_buffer.last()
            T_prev = self.T_buffer.last()
            P_prev = self.get_proj_mat(R_prev, T_prev)

            # Update current R and T estimates
            R_current = np.dot(R_cam, R_prev)
            T_current = np.dot(R_current.transpose(), T_cam) + T_prev
            self.R_buffer.push(R_current)
            self.T_buffer.push(T_current)

            # Get current projection matrix
            P_current = self.get_proj_mat(R_current, T_current)


            # Triangulate points to estimate 3D position in camera frame
            points_4d = cv2.triangulatePoints(P_prev, P_current, feature_matches[0], feature_matches[1])

            # Convert homogeneous coordinates to 3D coordinates
            points_3d = points_4d[:, :3]
            plot = gl.GLScatterPlotItem(pos=points_3d, color=pg.glColor('r'))
            self.plot_window.addItem(plot)
            self.plot_window.show()
            pg.QtGui.QApplication.exec_()


    def compute_RT(self, frame, show_features=False, feature_matches=None):
        if feature_matches is None:
            # Get feature matches
            prev_features, features = self.klt.get_feature_matches(frame, self.img_type,
                                                               motion_thresh=self.feature_motion_threshold)
        else:
            prev_features, features = feature_matches

        if show_features:
            display_features(frame, features)

        if len(features) < self.min_inliers:
            return None, None

        E, mask = cv2.findEssentialMat(prev_features, features, self.K,
                                       method=cv2.RANSAC, threshold=self.ransac_thresh, prob=self.ransac_prob)
        inliers = np.sum(mask)
        if inliers < self.min_inliers:
            return None, None
        ret, R, T, mask = cv2.recoverPose(E, prev_features, features, self.K, mask=mask)

        # Flip T and rotate it into the previous frame
        T = -np.dot(np.transpose(R),T).reshape(3,1)

        return R, T

    def get_proj_mat(self, R, T):
        P = np.hstack((R, np.reshape(T, (3, 1))))
        return np.dot(self.K,P)