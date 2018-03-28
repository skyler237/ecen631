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
        self.window_size = 2
        self.frame_buffer = FrameBuffer(self.window_size)

        # Initialize KLT tracker
        self.klt = KLTTracker(max_features=1000, min_features=500, dt=1.0 / 30.0, img_size=self.img_size,
                              img_type=self.img_type)

        # RANSAC params
        self.ransac_thresh = 0.1
        self.ransac_prob = 0.999

        # Other params
        self.min_inliers = 50
        self.omega_thresh = math.radians(0.1)
        self.feature_motion_threshold = 0.0 #px
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

        # Display variables and params
        self.points = []
        # 3D scatter plot
        self.app = pg.mkQApp()
        self.plot_window = gl.GLViewWidget()
        self.plot_window.addItem(gl.GLGridItem())
        self.plot_window.addItem(gl.GLAxisItem())
        self.scatter_plot = gl.GLScatterPlotItem()
        self.plot_window.addItem(self.scatter_plot)
        # 2D grid
        self.px_scale = 0.2 # m^2
        grid_width = 100.0 # m
        self.grid_size = int(grid_width/self.px_scale) # px
        self.grid = np.zeros((self.grid_size, self.grid_size), np.uint8)
        self.grid_decay_rate = 0.9
        self.grid_add_val = 20
        self.display_scale = int(1000*self.px_scale/grid_width)
        self.display_px = np.ones((self.display_scale, self.display_scale))


    def get_3d_points(self, frame, R_truth=None, T_truth=None):
        self.frame_buffer.add_frame(frame)
        if self.frame_buffer.cnt() == 1:
            # Initialize the feature tracker
            self.klt.get_feature_matches(frame,self.img_type, self.feature_motion_threshold)
        else:
            # Get feature matches
            feature_matches = self.klt.get_feature_matches(frame, self.img_type,
                                                                   motion_thresh=self.feature_motion_threshold)

            # Get previous projection matrix
            R_prev = self.R_buffer.last()
            T_prev = self.T_buffer.last()
            P_prev = self.get_proj_mat(R_prev, T_prev)

            # Switch for using true rotation and translation
            if R_truth is None or T_truth is None:
                R_cam, T_cam = self.compute_RT(frame, feature_matches=feature_matches)
                if R_cam is None or T_cam is None:
                    return # skip iteration if we don't get good data

            # Update current R and T estimates
            if R_truth is not None:
                R_current = np.dot(self.R_cam2body.transpose(), R_truth)
            else:
                R_current = np.dot(R_cam, R_prev)

            if T_truth is not None:
                T_current = np.dot(self.R_cam2body.transpose(), T_truth)
            else:
                # TODO: scale the translation here
                T_current = np.dot(R_current.transpose(), T_cam) + T_prev

            self.R_buffer.push(R_current)
            self.T_buffer.push(T_current)

            # Get current projection matrix
            P_current = self.get_proj_mat(R_current, T_current)


            # Triangulate points to estimate 3D position in camera frame
            points_4d = cv2.triangulatePoints(P_prev, P_current, feature_matches[0], feature_matches[1])

            # Convert homogeneous coordinates to 3D coordinates
            points_3d = np.dot(self.R_cam2body, points_4d[:3, :]/points_4d[3,:]).transpose()
            self.display_points(points_3d)

    def display_points(self, points, dim=2):
        if dim == 2:
            # Decay existing points in the grid
            self.grid = np.multiply(self.grid, self.grid_decay_rate).astype(np.uint8)

            # Compute pixel locations of new points
            points_2d = points[:, :2]
            px_points = np.floor_divide(points_2d, self.px_scale) + np.array([self.grid_size/2, self.grid_size/2]).astype(np.uint32)

            # Alter the grid
            for x,y in px_points.astype(np.uint32):
                if x in range(0,self.grid_size) and y in range(0,self.grid_size):
                    self.grid[x,y] = np.clip(self.grid[x,y] + self.grid_add_val, 0, 255)

            # Resize and rotate the grid

            # grid_display = cv2.resize(self.grid, tuple(np.multiply(np.shape(self.grid),self.display_scale)))
            grid_display = np.kron(self.grid, self.display_px)
            cv2.imshow("2D Grid", grid_display)
            cv2.waitKey(1)

        elif dim == 3:
            self.points.extend(points)
            self.scatter_plot.setData(pos=np.array(self.points), color=pg.glColor('r'), size=1)
            self.scatter_plot.update()
            self.plot_window.show()
            self.app.exec_()


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
