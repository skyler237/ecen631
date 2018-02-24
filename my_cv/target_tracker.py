from IPython.core.debugger import set_trace
import sys
import cv2
import numpy as np
import math
from my_cv import cv_utils
from my_cv.multi_image import MultiImage
from my_cv.filter import Filter
from my_cv.feature_tracker import KLTTracker, BGSubtractionTracker, MeanShiftTracker

############ Main target tracker ############
class TargetTracker:
    def __init__(self, tracker_type="KLT"):
        dt = 1/30.0
        # Initialize tracker
        tracker_types = {"KLT": KLTTracker,
                         "MeanShift": MeanShiftTracker,
                         "BGSubtraction": BGSubtractionTracker}
        self.tracker_type = tracker_types[tracker_type]
        self.tracker = self.tracker_type(dt=dt)
        self.roi_width = 0
        self.roi_height = 0

        # Initialize filter
        self.filter_model = Filter.CONST_ACCEL
        self.filter = Filter(self.filter_model, dt=dt)

    def track_targets(self, frame):
        measurements = self.tracker.get_measurements(frame)
        # Code here: Fix this problem
        # if np.size(measurements) > 4:
        meas = self.weighted_average(measurements)
        # else:
            # meas = measurements
        self.filter.correct(meas)
        filtered_meas = self.filter.predict()
        # FIXME: Try skipping the filter and feeding back in the value!
        # filtered_meas = meas
        region_center = np.copy([filtered_meas[0], filtered_meas[1]])
        # region_center = [int(feature_point[0][0]), int(feature_point[0][1])]
        self.tracker.set_roi(frame, *region_center, self.roi_width, self.roi_height, center=True)
        self.tracker.display('Target Tracker')

    def weighted_average(self, measurements):
        # Reshape the measurements to be column vectors
        measurements = np.reshape(measurements, (len(measurements),4,1))

        ### Find the mahalanobis distance of each measurement
        # Get innovation cov matrix
        S = self.filter.get_S_matrix()
        # Get current estimated state (only use pos and vel)
        xhat = self.filter.kalman.statePre[0:4]

        # Get mahalanobis distances
        D = np.array([[math.sqrt(np.matmul(np.matmul(np.transpose(y - xhat),np.linalg.inv(S)),(y - xhat)))] for y in measurements])
        D[np.where(D == 0)] = np.finfo(float).eps # convert zeros to smallest numbers (avoid divide by zero error)
        V = np.array([[np.linalg.norm(y[2:4])]for y in measurements])

        # Define cost
        kv = 5.0
        kd = 3.0
        C = kd*1.0/D + kv*V

        weighted_meas = np.array([c*m for c,m in zip(C,measurements)])
        weighted_avg = np.sum(weighted_meas,0)/np.sum(C)

        return weighted_avg


    def select_roi(self, frame):
        region = cv2.selectROI(frame)
        self.roi_width = region[2]
        self.roi_height = region[3]
        self.tracker.set_roi(frame, *region, center=False)

    def reset(self):
        self.tracker = self.tracker_type()
        self.filter = Filter(self.filter_model)
