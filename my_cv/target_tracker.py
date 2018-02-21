from IPython.core.debugger import set_trace
import sys
import cv2
import numpy as np
import math
from my_cv import cv_utils
from my_cv.multi_image import MultiImage

############ Main target tracker ############
class TargetTracker:
    def __init__(self, tracker_type="KLT"):
        dt = 1/30.0
        # Initialize tracker
        tracker_types = {"KLT": KLTTracker,
                         "MeanShift": MeanShiftTracker,
                         "BGSubtraction": BGSubtractionTracker}
        self.tracker_type = tracker_types[tracker_type]
        self.tracker = self.tracker_type(max_features=40, dt=dt)
        self.roi_width = 0
        self.roi_height = 0

        # Initialize filter
        self.filter_model = Filter.CONST_ACCEL
        self.filter = Filter(self.filter_model, dt=dt)

    def track_targets(self, frame):
        measurements = self.tracker.get_measurements(frame)
        avg_meas = self.weighted_average(measurements)
        self.filter.correct(avg_meas)
        filtered_meas = self.filter.predict()
        region_center = np.copy([filtered_meas[0], filtered_meas[1]])
        # region_center = [int(feature_point[0][0]), int(feature_point[0][1])]
        self.tracker.set_roi(*region_center, self.roi_width, self.roi_height, center=True)
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
        self.tracker.set_roi(*region, center=False)

    def reset(self):
        self.tracker = self.tracker_type()
        self.filter = Filter(self.filter_model)

############ Kalman Filter Wrapper ############
class Filter:
    CONST_VEL = 1
    CONST_ACCEL = 2
    CONST_JERK = 3
    def __init__(self, model_type=CONST_VEL, dt=1/30.0):
        self.dt = dt
        self.model_type = model_type

        self.meas_dim = 4
        self.sigma_R  = 30.0
        if model_type == self.CONST_VEL:
            self.state_dim = 4
            self.sigma_Q = 5.0
        elif model_type == self.CONST_ACCEL:
            self.state_dim = 6
            self.sigma_Q = 1.0
        elif model_type == self.CONST_JERK:
            self.state_dim = 8
            self.sigma_Q = 0.5

        data_type = np.float32
        self.kalman = cv2.KalmanFilter(self.state_dim,self.meas_dim)

        # Matrix definitions
        self.kalman.measurementMatrix   = self.get_C_matrix().astype(data_type)
        self.kalman.transitionMatrix    = self.get_A_matrix().astype(data_type)
        self.kalman.processNoiseCov     = self.get_Q_matrix().astype(data_type)
        self.kalman.measurementNoiseCov = self.sigma_R*np.eye(self.meas_dim).astype(data_type)

        # Initial conditions
        P_init = self.sigma_R # We use the first measurement to initialize the state
        self.kalman.errorCovPost = P_init*np.eye(self.state_dim).astype(data_type)
        self.kalman.errorCovPre = self.kalman.errorCovPost
        self.state_init = False

    def get_A_matrix(self):
        n = int(self.state_dim/2)
        A = np.eye(n)
        for row in range(0,n):
            for col in range(row+1,n):
                A[row,col] = self.dt**(col-row)
        return np.kron(A, np.eye(2))

    def get_Q_matrix(self):
        dt = self.dt
        if self.model_type == self.CONST_VEL:
            Q = np.array([[1.0/3*dt**3, 1.0/2*dt**2],
                          [1.0/2*dt**2,       dt   ]])
        elif self.model_type == self.CONST_ACCEL:
            Q = np.array([[1.0/20*dt**5, 1.0/8*dt**4, 1.0/6*dt**3],
                          [1.0/8 *dt**4, 1.0/3*dt**3, 1.0/2*dt**2],
                          [1.0/6 *dt**3, 1.0/2*dt**2,       dt   ]])
        elif self.model_type == self.CONST_JERK:
            Q = np.array([[1.0/252*dt**7, 1.0/72*dt**6, 1.0/30*dt**5, 1.0/24*dt**4],
                          [1.0/72 *dt**6, 1.0/20*dt**5, 1.0/8 *dt**4, 1.0/6 *dt**3],
                          [1.0/30 *dt**5, 1.0/8 *dt**4, 1.0/3 *dt**3, 1.0/2 *dt**2],
                          [1.0/24 *dt**4, 1.0/6 *dt**3, 1.0/2 *dt**2,        dt   ]])
        return self.sigma_Q*np.kron(Q, np.eye(2))

    def get_C_matrix(self):
        I = np.eye(self.meas_dim)
        zeros = np.zeros((self.meas_dim, self.state_dim - self.meas_dim))
        C = np.hstack((I,zeros))

        return C

    def get_S_matrix(self):
        P = self.kalman.errorCovPre
        C = self.kalman.measurementMatrix
        R = self.kalman.measurementNoiseCov

        S = np.matmul(np.matmul(C,P),C.transpose()) + R
        return S

    def predict(self):
        return self.kalman.predict()

    def correct(self, meas):
        # Only run the correction if there is a measurement
        if len(meas) == 0:
            return

        # Initialize state with first measurement
        if not self.state_init:
            # set_trace()
            zeros = np.zeros((self.state_dim - self.meas_dim,1))
            init_state = np.vstack((meas,zeros))
            self.kalman.statePost = init_state.astype(np.float32)
            self.kalman.statePre = init_state.astype(np.float32)
            self.state_init = True

        return self.kalman.correct(meas.astype(np.float32))


############ Individual tracker implementations ############
class Tracker:
    ''' Template class for tracking in an image
    '''
    def __init__(self, dt=1/30.0):
        # Default image parameters
        self.default_img_size = (480,720,3)
        self.default_img_type = 'bgr'

        # Default variables
        self.dt = dt
        self.tracks_mask = np.zeros(self.default_img_size, dtype=np.uint8)
        self.display_img = np.zeros(self.default_img_size, dtype=np.uint8)
        # Create empty grayscale region of interest mask
        self.roi = np.zeros((self.default_img_size[0], self.default_img_size[1]), dtype=np.uint8)
        self.roi_pos = [0,0]
        self.multi_image = MultiImage(2,1)

        # For testing:
        self.est_features = []

    def get_measurements(self, frame):
        pass

    def set_roi(self, region_x, region_y, region_width, region_height, center=False):
        # Blank out roi
        self.roi = np.zeros((self.default_img_size[0], self.default_img_size[1]), dtype=np.uint8)
        if center:
            region_x -= int(region_width/2)
            region_y -= int(region_height/2)
        self.roi_pos = (region_x, region_y)
        cv2.rectangle(self.roi, self.roi_pos, (region_x+region_width,region_y+region_height), color=255, thickness=cv2.FILLED)
        cv2.rectangle(self.display_img, self.roi_pos, (region_x+region_width,region_y+region_height), color=[0,0,255], thickness=2)

    def display(self, img_name='Tracker'):
        # self.multi_image.set_image(self.display_img,0,0)
        # self.multi_image.set_image(self.roi, 1,0)
        # cv2.imshow(img_name, self.multi_image.get_display())
        cv2.imshow(img_name, self.display_img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            sys.exit()


class KLTTracker(Tracker):
    def __init__(self, max_features=1, dt=1/30.0):
        # Inherit fram Tracker init
        super().__init__(dt=dt)

        # Initialize parameters
        self.feature_params = dict( maxCorners = max_features,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

        self.lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Initialize data and variables
        self.initialized = False
        self.prev_frame = np.zeros(self.default_img_size)
        self.prev_gray = np.zeros((self.default_img_size[0], self.default_img_size[1]))
        self.prev_meas = np.zeros((1,4,1))
        self.colors = np.random.randint(0,255,(max_features,3))

    def initialize_features(self, frame):
        self.prev_frame = frame
        self.prev_gray = cv_utils.get_gray(frame)
        self.initialized = True

    def get_measurements(self, frame):
        if not self.initialized:
            self.initialize_features(frame)

        gray = cv_utils.get_gray(frame)

        # Select good features in the roi
        self.features = cv2.goodFeaturesToTrack(gray, mask=self.roi, **self.feature_params)
        if self.features is None:
            return self.prev_meas

        # Calculate optical flow
        self.new_features, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.features, None, **self.lk_params)

        # Throw away bad points
        self.new_features = self.new_features[status==1]
        self.features = self.features[status==1]

        # Calculate optical flow velocity
        u = (self.new_features - self.features)/self.dt
        u = np.reshape(u, (len(u), 2,1))

        # Update display
        self.update_display(frame)


        # Update previous data
        self.prev_gray = gray.copy()
        self.features = self.new_features.reshape(-1,2,1)

        # Concatenate positions and velocities to create measurements
        meas = np.hstack((self.features,u))

        self.prev_meas = meas

        return meas

    def update_display(self, frame):
        for i,(new,old) in enumerate(zip(self.new_features,self.features)):
            a,b = new.ravel()
            c,d = old.ravel()
            self.tracks_mask = cv2.line(self.tracks_mask, (a,b),(c,d), (0,255,0), 2)
            frame = cv2.circle(frame,(a,b),2,(0,255,0),-1)
        for i,(feat) in enumerate(self.est_features):
            a,b = feat.ravel()
            frame = cv2.circle(frame,(a,b),2,(0,0,255),-1)
        self.display_img = cv2.add(frame, self.tracks_mask)


class MeanShiftTracker(Tracker):
    pass

class BGSubtractionTracker(Tracker):
    pass
