import sys
rm_python2 = []
for p in sys.path:
    if p.find('python2') != -1:
        rm_python2.append(p)
for p in rm_python2:
    sys.path.remove(p)

import cv2
import numpy as np

##### Common utility functions #####
def get_gray(img, img_type='bgr'):
    if img_type == 'bgr':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img_type == 'hsv':
        return cv2.cvtColor(img, cv2.COLOR_HSV2GRAY)
    elif img_type == 'rgba':
        return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    else:
        print("Invalid image type: {0}".format(img_type))

class OpticalFlow():
    def __init__(self):
        self.prev_gray = None
        self.initialized = False

        self.lk_params = dict( winSize  = (15,15),
                               maxLevel = 2,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



        self.feature_params = dict( maxCorners = 100,
                                qualityLevel = 0.3,
                                minDistance = 7,
                                blockSize = 7 )

        self.default_size = (512,512,4)
        self.dt = 1.0/30.0

        # Define a point grid
        pixels_per_point = 128
        img_height = self.default_size[0]
        img_width = self.default_size[1]
        self.grid = np.array([[[np.float32(i),np.float32(j)]] for i in range(0,img_height)[::pixels_per_point]
                            for j in range(0,img_width)[::pixels_per_point]])

        self.p0 = self.grid
        self.mask = np.zeros((512,512,4))

        self.color = np.random.randint(0,255,(np.shape(self.grid)[0],3))

        self.display = np.zeros(self.default_size)

        self.u = np.zeros_like(self.grid)


    def compute_optical_flow(self, frame):
        # Initialize prev data, if needed
        if not self.initialized:
            self.prev_gray = get_gray(frame, img_type='rgba')
            # self.p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask = None, **self.feature_params)
            # self.p0 = self.grid
            self.mask = np.zeros_like(frame)
            self.initialized = True

        # Get gray images
        gray = get_gray(frame, img_type='rgba')


        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, **self.lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = self.p0[st==1]

        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            self.mask = cv2.line(self.mask, (a,b),(c,d), self.color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,self.color[i].tolist(),-1)
        img = cv2.add(frame,self.mask)

        self.prev_gray = gray.copy()
        # self.p0 = good_new.reshape(-1,1,2)

        self.display = img

        # Get static optical flow
        self.u = (good_new - good_old)/self.dt

        return self.u

    def display_image(self):
        cv2.imshow('Optical Flow',self.display)
        cv2.waitKey(1)
