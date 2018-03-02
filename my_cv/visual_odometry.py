from IPython.core.debugger import set_trace
from my_cv.cv_utils import CamParams
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
            self.default_img_size = (512,512,4)
            self.default_img_type = 'rgba'
            self.f = 512/2
            self.K = np.array([[self.f, 0.0,    512/2],
                               [0.0,    self.f, 512/2],
                               [0.0,    0.0,    1.0]])
        else:
            # Extract camera params
            cam_params = CamParams(cam_param_file)
            self.K = cam_params.K
            self.default_img_size = (cam_params.width,cam_params.height,3)
            self.default_img_type = 'bgr'
            self.f = (self.K[0][0] + self.K[1][1])/2.0 # Take the average of fx and fy


        # Initialize KLT tracker
        self.klt = KLTTracker(max_features=300, dt=1.0/30.0)


    def compute_RT(self, frame):
        # Get feature matches
        # REVIEW: Are these flipped? why?
        features, prev_features = self.klt.get_feature_matches(frame)

        # REVIEW: How much do I need to tune RANSAC? Also, do I want to mask this?
        E, mask = cv2.findEssentialMat(prev_features, features, self.K, method=cv2.RANSAC)
        ret, R, T, mask = cv2.recoverPose(E, prev_features, features, self.K, mask=mask)

        euler = np.degrees(transforms3d.euler.mat2euler(R, 'rxyz'))

        return R, T
