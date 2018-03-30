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
            self.img_size = (512, 512, 4)
            self.img_type = 'rgba'
            self.f = 512 / 2
            self.K = np.array([[self.f, 0.0, 512 / 2],
                               [0.0, self.f, 512 / 2],
                               [0.0, 0.0, 1.0]])
        else:
            # Extract camera params
            cam_params = CamParams(cam_param_file)
            self.K = cam_params.K
            self.img_size = (cam_params.width, cam_params.height, 3)
            self.img_type = 'bgr'
            self.f = (self.K[0][0] + self.K[1][1]) / 2.0  # Take the average of fx and fy

        # Initialize frame buffer
        self.window_size = 2
        self.frame_buffer = FrameBuffer(self.window_size)

        # Initialize KLT tracker
        self.klt = KLTTracker(max_features=2000, min_features=1000, dt=1.0 / 30.0, img_size=self.img_size,
                              img_type=self.img_type)
        self.roi_height = 100

        # RANSAC params
        self.ransac_thresh = 0.01
        self.ransac_prob = 0.9999

        # Other params
        self.min_inliers = 50
        self.omega_thresh = math.radians(0.01)
        self.feature_motion_threshold = 0.1  # px
        self.euler_threshold = math.radians(10.0)
        self.dt = 1.0/30.0

        # Initialize state buffers
        self.T_buffer = Buffer(self.window_size)
        T_init = np.zeros((3, 1))  # estimated position
        self.T_buffer.push(T_init)

        self.R_buffer = Buffer(self.window_size)
        R_init = np.eye(3)  # estimated rotation (matrix)
        self.R_buffer.push(R_init)

        self.R_prev_cam = R_init
        self.T_prev_cam = T_init
        self.initialized = False

        self.R_cam2body = np.array([[0., 0., 1.],
                                    [1., 0., 0.],
                                    [0., -1., 0.]])
        self.P0 = np.dot(self.K, np.hstack((np.eye(3), np.ones((3, 1)))))

        # 3D points pruning params
        # self.min_y = -1.0 # m
        # self.max_y = 0.0 # m
        # self.min_z = 0 # m
        # self.max_z = 1000 # m
        self.min_y = -np.infty  # m
        self.max_y = np.infty  # m
        self.min_z = 0  # m
        self.max_z = 500  # m

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
        self.px_scale = 0.5  # m^2
        grid_width = 40.0  # m
        self.grid_size = int(grid_width / self.px_scale)  # px
        self.grid = np.zeros((self.grid_size, self.grid_size), np.uint8)
        self.grid_decay_rate = 0.99
        self.grid_add_val = 10
        self.display_scale = int(1000 * self.px_scale / grid_width)
        self.display_shape = (self.grid_size * self.display_scale,) * 2
        self.display_px = np.ones((self.display_scale, self.display_scale))

    def get_3d_points2(self, frame, body_vel, R_truth=None, T_truth=None, use_truth=False):
        # if R_truth is not None and T_truth is not None:
        #     use_truth = True
        # else:
        #     use_truth = False

        if not self.initialized:
            # Initialize the feature tracker
            self.klt.set_roi(frame, int(self.img_size[0] / 2), int(self.img_size[1] / 2),
                             self.img_size[1], self.roi_height, center=True, bg_sub=False)
            self.klt.get_feature_matches(frame, self.img_type, self.feature_motion_threshold)
            self.initialized = True
        else:

            # Get feature matches
            feature_matches = self.klt.get_feature_matches(frame, self.img_type,
                                                           motion_thresh=self.feature_motion_threshold)
            display_features(frame, feature_matches[1])
            if feature_matches[0] is None or feature_matches[1] is None:
                return  # skip this iteration if no feature matches

            # Get Rotation and translation info between frames
            if use_truth:
                R_truth_cam = np.dot(self.R_cam2body.transpose(), np.dot(R_truth, self.R_cam2body))
                T_truth_cam = np.dot(self.R_cam2body.transpose(), T_truth)

                # R_step = R(k-1 -> k) = R(0 -> k)*R(k-1 -> 0)
                R_step = np.dot(R_truth_cam, self.R_prev_cam.transpose())
                # T_step = T(k -> k-1 in k) = -R(0 -> k-1)*[T(0->k in 0) - T(0->k-1 in 0)]
                T_step = np.dot(R_truth_cam, self.T_prev_cam.reshape(3, 1) - T_truth_cam.reshape(3, 1))
                # print("T_diff = {0}".format(self.T_prev_cam.reshape(3,1) - T_truth_cam.reshape(3,1)))

                R_current_cam = np.copy(R_truth_cam)
                T_current_cam = np.copy(T_truth_cam)
            else:
                R_step, T_step = self.compute_RT(frame, feature_matches=feature_matches)
                if R_step is None or T_step is None:
                    return # skip iteration
                # Accumulate R and T
                R_current_cam = R_step.dot(self.R_prev_cam)
                cam_vel = self.R_cam2body.T.dot(body_vel)
                scale = np.linalg.norm(cam_vel)*self.dt
                T_current_cam = (-R_current_cam.T.dot(T_step*scale) + self.T_prev_cam)


            # print("T_step = {0}".format(T_step))
            # print("R_step = {0}".format(transforms3d.euler.mat2euler(R_step)))
            P1 = self.get_proj_mat(R_step, T_step)

            # Triangulate points
            points_4d = cv2.triangulatePoints(self.P0, P1, feature_matches[0], feature_matches[1])
            # print("Average 4d point = {0}".format(np.average(points_4d,1)))

            # Convert homogeneous coordinates to 3D coordinates
            points_3d_cam = points_4d[:3, :] / points_4d[3, :]
            # print("Average cam point = {0}".format(np.average(points_3d_cam,1)))

            # Prune points
            pruned_points_cam = self.prune_3d_points(points_3d_cam)

            # Transform back to initial frame
            # X(0->x in 0) = R(k-1 -> 0)*X(k-1 -> x in k-1) + T(0->k-1 in 0)
            points_3d_cam0 = np.dot(self.R_prev_cam.transpose(), pruned_points_cam) + self.T_prev_cam.reshape(3, 1)

            # Convert to NED
            points_3d_world = np.dot(self.R_cam2body, points_3d_cam0).transpose()

            # pos = np.dot(self.R_cam2body, T_current_cam.reshape(3,1))[:2]
            pos = T_truth[:2]
            print("Pos = {0}".format(pos))
            self.R_prev_cam = R_current_cam
            self.T_prev_cam = T_current_cam
            self.display_points(points_3d_world, pos)

    def get_3d_points0(self, frame, R_truth=None, T_truth=None):
        self.frame_buffer.add_frame(frame)
        if self.frame_buffer.cnt() == 1:
            # Initialize the feature tracker
            self.klt.get_feature_matches(frame, self.img_type, self.feature_motion_threshold)
        else:
            # Get feature matches
            feature_matches = self.klt.get_feature_matches(frame, self.img_type,
                                                           motion_thresh=self.feature_motion_threshold)
            if feature_matches[0] is None:
                return  # skip this iteration if no feature matches

            # display_features(frame, feature_matches[1])
            # TODO: Try cv2.correctmatches function

            # Get previous projection matrix
            R_prev = self.R_buffer.peek(0)
            T_prev = self.T_buffer.peek(0)
            P_prev = self.get_proj_mat(R_prev, T_prev)

            # Switch for using true rotation and translation
            if R_truth is None or T_truth is None:
                R_cam, T_cam = self.compute_RT(frame, feature_matches=feature_matches)
                if R_cam is None or T_cam is None:
                    return  # skip iteration if we don't get good data

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
            # R_step = R(k-1 -> k) = R(0 -> k)*R(k-1 -> 0)
            R_step = np.dot(R_current, R_prev.transpose())
            # T_step = T(k -> k-1 in k-1) = -R(0 -> k-1)*[T(0->k in 0) - T(0->k-1 in 0)]
            T_step = -np.dot(R_prev, (T_current.reshape(3, 1) - T_prev.reshape(3, 1)))
            P_step = self.get_proj_mat(R_step, T_step)
            print("T_current = {0}".format(T_current))
            print("T_step = {0}".format(T_step))
            print("R_step = {0}".format(transforms3d.euler.mat2euler(R_step)))
            # P_current = self.get_proj_mat(R_current, T_current)

            # Triangulate points to estimate 3D position in camera frame
            try:
                points_4d = cv2.triangulatePoints(self.P0, P_step, feature_matches[0], feature_matches[1])
            except:
                return

            # Convert homogeneous coordinates to 3D coordinates
            points_3d_cam = points_4d[:3, :] / points_4d[3, :]
            print("Average cam point = {0}".format(np.average(points_3d_cam, 1)))

            # Prune points
            pruned_points_cam = self.prune_3d_points(points_3d_cam)
            # pruned_points_cam = np.copy(points_3d_cam)

            # Translate points
            # X(0->x in 0) = R(k-1 -> 0)*X(k-1 -> x in k-1) + T(0->k-1 in 0)
            points_3d_cam0 = np.dot(R_prev.transpose(), pruned_points_cam) + T_prev.reshape(3, 1)
            # points_3d_cam0 = pruned_points_cam + T_current.reshape(3,1)

            # Convert to NED
            points_3d_world = np.dot(self.R_cam2body, points_3d_cam0).transpose()

            pos = np.dot(self.R_cam2body, T_current.reshape(3,1))[:2]
            self.display_points(points_3d_world, pos)

    def prune_3d_points(self, points):
        # convert points to row vectors for easier boolean indexing
        points = np.copy(points).reshape(-1, 3)
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Limit altitude (negative y direction)
        good_y = np.logical_and([y >= self.min_y], [y <= self.max_y])

        # Limit z_axis
        good_z = np.logical_and([z >= self.min_z], [z <= self.max_z])

        good_pts = np.logical_and(good_y, good_z).reshape(points.shape[0])

        pruned_points = points[good_pts]

        # Convert points back to column vectors
        return pruned_points.reshape(3, -1)

    def display_points(self, points, pos, dim=2):
        if dim == 2:
            # Decay existing points in the grid
            self.grid = np.multiply(self.grid, self.grid_decay_rate).astype(np.uint8)

            # Compute pixel locations of new points
            points_2d = points[:, :2]
            px_points = np.floor_divide(points_2d, self.px_scale) + np.array(
                [self.grid_size / 2, self.grid_size / 2]).astype(np.uint32)

            # Alter the grid
            for x, y in px_points.astype(np.uint32):
                if x in range(0, self.grid_size) and y in range(0, self.grid_size):
                    px_x = (self.grid_size - 1) - x
                    px_y = y
                    self.grid[px_x, px_y] = np.clip(self.grid[px_x, px_y] + self.grid_add_val, 0, 255)

            # Convert to color for viewing
            grid_display = cv2.cvtColor(self.grid, cv2.COLOR_GRAY2BGR)

            pos_px = np.floor_divide(pos, self.px_scale) + np.array([self.grid_size / 2, self.grid_size / 2]).astype(
                np.uint32).reshape(np.shape(pos))
            pos_x = int(pos_px[0])
            pos_y = int(pos_px[1])
            if pos_x in range(0, self.grid_size) and pos_y in range(0, self.grid_size):
                px_x = pos_y
                px_y = (self.grid_size - 1) - pos_x
                grid_display[px_y, px_x, :] = [0, 0, 255]

            # Resize and rotate the grid
            # grid_display = np.kron(grid_display, self.display_px)
            grid_display = cv2.resize(grid_display, (0, 0), fx=self.display_scale, fy=self.display_scale,
                                      interpolation=cv2.INTER_NEAREST)
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
        # T = -np.dot(np.transpose(R), T).reshape(3, 1)

        return R, T

    def get_proj_mat(self, R, T):
        P = np.hstack((R, np.reshape(T, (3, 1))))
        return np.dot(self.K, P)
