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
        self.default_img_size = (480,720,3)
        self.default_img_type = 'bgr'

        self.prev_gray = np.zeros((self.default_img_size[0], self.default_img_size[1]), np.uint8)

        # Default variables
        self.initialized = False
        self.tracks_mask = np.zeros(self.default_img_size, dtype=np.uint8)
        self.display_img = np.zeros(self.default_img_size, dtype=np.uint8)

        # Create empty grayscale region of interest mask
        self.roi_mask = np.zeros((self.default_img_size[0], self.default_img_size[1]), dtype=np.uint8)
        self.roi = (0,0,0,0)
        self.multi_image = MultiImage(2,1)

        # For testing:
        self.est_features = []

    def set_roi(self, frame, region_x, region_y, region_width, region_height, center=False):
        if center:
            region_x -= int(region_width/2)
            region_y -= int(region_height/2)
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
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

        self.lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Initialize data and variables
        self.dt = dt
        self.prev_meas = np.zeros((1,4,1))
        self.colors = np.random.randint(0,255,(max_features,3))

        # Background subtraction
        self.prev_bg = np.zeros((self.default_img_size[0], self.default_img_size[1]), np.uint8)
        self.bgsub = cv_utils.BackgrondSubtractor(display=True)

    def initialize_features(self, frame):
        self.prev_gray = cv_utils.get_gray(frame)
        self.initialized = True

    def get_measurements(self, frame):
        if not self.initialized:
            self.initialize_features(frame)

        gray = cv_utils.get_gray(frame)

        # Select good features in the roi
        self.features = cv2.goodFeaturesToTrack(gray, mask=self.roi_mask, **self.feature_params)
        if self.features is None:
            self.features = []
            self.new_features = []
            # Update display
            self.update_display(frame)
            # Update previous data
            self.prev_gray = gray.copy()
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

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    def set_roi(self, frame, region_x, region_y, region_width, region_height, center=False):
        # Overwrite width and height with camshift values (if initialized)
        if self.initialized:
            region_width = self.roi_width
            region_height = self.roi_height
        else:
            self.roi_width = region_width
            self.roi_height = region_height
        super().set_roi(frame, region_x, region_y, region_width, region_height, center)

        if not self.initialized:
            self.roi_prev = self.roi
            col,row,w,h = self.roi
            # Crop the image
            roi_img = frame[row:row+h, col:col+w]
            # Convert to HSV
            hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
            # Threshold the HSV values
            mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
            # Compute a histogram
            self.roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            # Normalize the histogram
            cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)
            self.initialized = True

    def get_measurements(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)
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
