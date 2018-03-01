import numpy as np
import cv2
import math
import yaml

from my_cv.cv_utils import CamParams

class VO:
    def __init__(self, cam_param_file=None):
        if cam_param_file is None:
            # Default camera parameters -- Holodeck
            self.default_img_size = (512,512,4)
            self.default_img_type = 'rgba'
            self.f = 512/2
            self.K = np.array([[self.f, 0.0,    512/2],
                               [0.0,    self.f, 512/2],
                               [0.0,    0.0     1.0]])
        else:
            # Extract camera params
            cam_params = CamParams(cam_param_file)
            self.K = cam_params.K
            self.default_img_size = (cam_params.width,cam_params.height,3)
            self.default_img_type = 'bgr'
            self.f = (self.K[0][0] + self.K[1][1])/2.0 # Take the average of fx and fy



        self.prev_img = np.zeros(self.default_img_size)
        self.E = np.eye(3)

    def compute_essential_matrix(self, frame):
        pass

    def compute_RT(self, frame):
        pass
