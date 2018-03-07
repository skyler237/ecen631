from IPython.core.debugger import set_trace
import sys
import cv2
import numpy as np
import math
from my_cv import cv_utils
from my_cv.multi_image import MultiImage


############ Individual tracker implementations ############
class FeatureTracker:
    ''' Template class for tracking in an image
    '''
    def __init__(self):
        # Default image parameters
        self.default_img_size = (512,512,3)
        self.default_img_type = 'bgr'

        self.prev_gray = np.zeros((self.default_img_size[0], self.default_img_size[1]), np.uint8)

        # Default variables
        self.initialized = False
        self.tracks_mask = np.zeros(self.default_img_size, dtype=np.uint8)
        self.display_img = np.zeros(self.default_img_size, dtype=np.uint8)

        # Create empty grayscale region of interest mask
        self.roi_mask = np.ones((self.default_img_size[0], self.default_img_size[1]), dtype=np.uint8)*255
        self.roi = (0,0,0,0)
        self.multi_image = MultiImage(2,1)

        # For testing:
        self.est_features = []

    def set_roi(self, frame, region_x, region_y, region_width, region_height, center=False):
        if center:
            region_x -= (region_width/2)
            region_y -= (region_height/2)
        region_x, region_y, region_width, region_height = int(region_x), int(region_y), int(region_width), int(region_height)
        self.roi = (region_x, region_y, region_width, region_height)
        self.display_img = cv2.rectangle(frame, (region_x,region_y), (region_x+region_width,region_y+region_height), color=[0,0,255], thickness=2)

    def get_measurements(self, frame):
        pass

    def display(self, img_name='Tracker'):
        # self.multi_image.set_image(self.display_img,0,0)
        # self.multi_image.set_image(self.roi_mask, 1,0)
        # cv2.imshow(img_name, self.multi_image.get_display())
        cv2.imshow(img_name, self.display_img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            sys.exit()


class KLTTracker(FeatureTracker):
    def __init__(self, max_features=40, dt=1/30.0):
        # Inherit fram Tracker init
        super().__init__()

        # Initialize parameters
        self.feature_params = dict( maxCorners = max_features,
                       qualityLevel = 0.2,
                       minDistance = 9,
                       blockSize = 5 )

        self.lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Initialize data and variables
        self.dt = dt
        self.prev_meas = np.zeros((1,4,1))
        self.colors = np.random.randint(0,255,(max_features,3))

        # Background subtraction
        self.prev_bg = np.zeros((self.default_img_size[0], self.default_img_size[1]), np.uint8)
        self.bgsub = cv_utils.BackgroundSubtractor(display=True)

    def initialize_features(self, frame):
        self.prev_gray = cv_utils.get_gray(frame, self.default_img_type)
        self.initialized = True

    def get_feature_matches(self, frame, img_type='bgr'):
        if not self.initialized:
            self.initialize_features(frame)

        gray = cv_utils.get_gray(frame, self.default_img_type)

        # Select good features in the roi
        features = cv2.goodFeaturesToTrack(self.prev_gray, mask=self.roi_mask, **self.feature_params)

        # Calculate optical flow
        new_features, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, features, None, **self.lk_params)

        # Update previous data
        self.prev_gray = gray.copy()

        if new_features is None:
            return [None, None]
        else:
            # Throw away bad points
            new_features = new_features[status==1]
            features = features[status==1]
            return [features, new_features]

    def get_measurements(self, frame):
        self.features, self.new_features = self.get_feature_matches(frame)
        if self.features is None:
            self.features = []
            self.new_features = []
            # Update display
            self.update_display(frame)
            return self.prev_meas

        # Calculate optical flow velocity
        u = (self.new_features - self.features)/self.dt
        u = np.reshape(u, (len(u), 2,1))

        # Update display
        self.update_display(frame)

        # Concatenate positions and velocities to create measurements
        meas = np.hstack((self.new_features.reshape(-1,2,1),u))

        self.prev_meas = meas

        return meas

    def set_roi(self, frame, region_x, region_y, region_width, region_height, center=False):
        super().set_roi(frame, region_x, region_y, region_width, region_height, center)

        x,y,w,h = self.roi
        # Blank out roi
        self.roi_mask = np.zeros((self.default_img_size[0], self.default_img_size[1]), dtype=np.uint8)
        cv2.rectangle(self.roi_mask, (x,y), (x+w,y+h), color=255, thickness=cv2.FILLED)

        # Use background subtraction to isolate moving targets within roi
        fgmask = self.bgsub.get_fg(frame)
        # Get the intersection between the foreground and roi rectangle
        self.roi_mask = np.logical_and(fgmask, self.roi_mask).astype(np.uint8)*255

    def update_display(self, frame):
        frame = np.copy(frame)
        for i,(new,old) in enumerate(zip(self.new_features,self.features)):
            a,b = new.ravel()
            c,d = old.ravel()
            self.tracks_mask = cv2.line(self.tracks_mask, (a,b),(c,d), (0,255,0), 2)
            frame = cv2.circle(frame,(a,b),2,(0,255,0),-1)
        for i,(feat) in enumerate(self.est_features):
            a,b = feat.ravel()
            frame = cv2.circle(frame,(a,b),2,(0,0,255),-1)
        self.display_img = cv2.add(frame, self.tracks_mask)


class CamShiftTracker(FeatureTracker):
    def __init__(self, dt=1.0/30):
        # Inherit fram Tracker init
        super().__init__()

        self.initialized = False
        self.dt = dt
        self.roi_hist = None
        self.roi_prev = (0,0,0,0)
        self.roi_width = 0
        self.roi_height = 0
        self.roi_width_max = 50
        self.roi_height_max = 50
        self.roi_width_min = 20
        self.roi_height_min = 20

        # HSV threshold values
        self.hue_min = 10
        self.hue_max = 180
        self.sat_min = 0
        self.sat_max = 80
        self.val_min = 15
        self.val_max = 240
        # self.hue_min = 0
        # self.hue_max = 180
        # self.sat_min = 60
        # self.sat_max = 255
        # self.val_min = 32
        # self.val_max = 255

        # self.hist_channels = [0,1,2]
        # self.hist_bins = [self.hue_max - self.hue_min, self.sat_max - self.sat_min, self.val_max - self.val_min]
        # self.hist_ranges = [self.hue_min, self.hue_max, self.sat_min, self.sat_max, self.val_min, self.val_max]
        # self.hist_channels = [0,1]
        # self.hist_bins = [self.hue_max - self.hue_min, self.sat_max - self.sat_min]
        # self.hist_ranges = [self.hue_min, self.hue_max, self.sat_min, self.sat_max]
        self.hist_channels = [0]
        self.hist_bins = [self.hue_max]
        self.hist_ranges = [self.hue_min, self.hue_max]

        self.bgsub = cv_utils.BackgroundSubtractor(display=True)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    def set_roi(self, frame, region_x, region_y, region_width, region_height, center=False):
        # Overwrite width and height with camshift values (if initialized)
        if self.initialized:
            region_width = max(min(self.roi_width, self.roi_width_max), self.roi_width_min)
            region_height = max(min(self.roi_height, self.roi_height_max), self.roi_height_min)
        else:
            self.roi_width = region_width
            self.roi_height = region_height
        region_x = max(min(region_x, self.default_img_size[0]-region_width/2),region_width/2)
        region_y = max(min(region_y, self.default_img_size[1]-region_height/2),region_height/2)
        super().set_roi(frame, region_x, region_y, region_width, region_height, center)

        if not self.initialized:
            self.roi_prev = self.roi
            self.compute_histogram(frame)
            self.initialized = True

    def compute_histogram(self, frame):
        # Get current roi
        col,row,w,h = self.roi
        # Crop the image
        roi_img = frame[row:row+h, col:col+w]
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        # Threshold the HSV values
        mask = cv2.inRange(hsv_roi, (self.hue_min, self.sat_min, self.val_min),
                                    (self.hue_max, self.sat_max, self.val_max))

        # roi_fg = fg[row:row+h, col:col+w]
        # # Only consider moving targets
        # mask = np.logical_and(roi_fg,mask).astype(np.uint8)*255

        # Compute a histogram
        self.roi_hist = cv2.calcHist([hsv_roi],self.hist_channels,mask,
                            self.hist_bins,
                            self.hist_ranges)
        # Normalize the histogram
        cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)

    def get_measurements(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # REVIEW: Do we want to recompute at each timestep?
        self.compute_histogram(frame)
        # Get the moving targets (foreground)
        fg = self.bgsub.get_fg(frame)/255 # Scale down to zeros and ones
        hsv[:,:,0] = hsv[:,:,0]*fg
        dst = cv2.calcBackProject([hsv],self.hist_channels,self.roi_hist,self.hist_ranges,1)
        # apply cmashift to get the new location
        ret, new_roi = cv2.CamShift(dst, self.roi, self.term_crit)

        # Grab new width and height values
        self.roi_width = new_roi[2]
        self.roi_height = new_roi[3]

        # set_trace()
        # compute center
        x,y,w,h = new_roi
        center = np.array([[int(x+w/2)],
                           [int(y+h/2)]])
        # compute velocity
        x_prev, y_prev = self.roi_prev[0:2]

        vel = 1.0/self.dt * np.array([[float(x - x_prev)],
                                      [float(y - y_prev)]])

        meas = np.vstack((center,vel))
        self.roi_prev = self.roi
        return [meas]


class BGSubtractionTracker(FeatureTracker):
    pass
