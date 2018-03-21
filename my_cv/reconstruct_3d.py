from IPython.core.debugger import set_trace
from my_cv.cv_utils import *
from my_cv.feature_tracker import KLTTracker
import numpy as np
import transforms3d
import cv2
import math
import yaml

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
        self.frame_buffer = cv_utils.FrameBuffer(window_size)

        # Initialize state
        self.phat = np.array([0., 0., 0.])  # estimated position
        self.Rhat = np.eye(3)               # estimated rotation (matrix)

        self.R_cam2body = np.array([[0.,  0.,  1.],
                                    [1.,  0.,  0.],
                                    [0.,  1.,  0.]])

    def get_3d_points(self, frame):
        self.frame_buffer.add_frame(frame)
        if self.frame_buffer.cnt() == self.window_size:
            # Store frames in files
            image_files = []
            for i, img in enumerate(self.frame_buffer.get_frames()):
                filename = "img{0}.png".format(i)
                image_files.append(filename)
                cv2.imwrite(filename, img)

            # Call reconstruct function
