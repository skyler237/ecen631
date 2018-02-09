import sys
rm_python2 = []
for p in sys.path:
    if p.find('python2') != -1:
        rm_python2.append(p)
for p in rm_python2:
    sys.path.remove(p)

import cv2
import numpy as np
import math

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

def get_grid(origin, width, height, num_points):
    ''' Returns a grid of evenly spaced points
        origin: [x,y] pixel position of top left corner of region
        width: width of the region in pixels
        height: height of the region in pixels
        px_per_pt: pixel spacing between points
    '''
    px_per_pt = int(math.sqrt(width*height/num_points))

    offset_x = origin[0] + px_per_pt/2
    offset_y = origin[1] + px_per_pt/2
    grid = np.array([[[np.float32(x+offset_x),np.float32(y+offset_y)]] for y in range(0,height)[::px_per_pt]
                        for x in range(0,width)[::px_per_pt]])
    return grid

class FrameBuffer():
    def __init__(self, buffer_size):
        self.size = buffer_size
        self.buffer = []

    def add_frame(self, frame):
        self.buffer.insert(0, frame.copy())
        while len(self.buffer) > self.size:
            self.buffer.pop()

    def fill(self, frame):
        self.clear()
        for i in range(0,self.size):
            self.add_frame(frame)

    def pop(self):
        return self.buffer.pop()

    def peek(self, i):
        if i < self.size:
            return self.buffer[i]
        else:
            print("Invalid index: {0}".format(i))

    def set_size(self, size):
        self.size = size

    def cnt(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []


####################################################
############### Computer Vision Classes ############
####################################################

class OpticalFlow():
    def __init__(self, buffer_size=1):
        self.regions_p0 = {}
        self.regions_prev_gray = {}
        self.regions_frame_buffer = {}
        self.buffer_size = buffer_size

        self.lk_params = dict( winSize  = (15,15),
                               maxLevel = 2,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



        self.feature_params = dict( maxCorners = 100,
                                qualityLevel = 0.3,
                                minDistance = 7,
                                blockSize = 7 )

        self.default_size = (512,512,4)
        self.dt = 1.0/30.0

        # # Define a point grid
        # px_per_pt = 100
        # img_height = self.default_size[0]
        # img_width = self.default_size[1]
        # offset = px_per_pt/2
        # self.grid = np.array([[[np.float32(i+offset),np.float32(j+offset)]] for i in range(0,img_height)[::px_per_pt]
        #                     for j in range(0,img_width)[::px_per_pt]])
        #
        # self.p0 = self.grid
        # self.u = np.zeros_like(self.grid)

        self.num_points = 300
        # self.color = np.random.randint(0,255,(np.shape(self.grid)[0],3))
        self.color = [0,0,255]

        self.display = np.zeros(self.default_size)
        self.display_init = False

    def compute_optical_flow(self, frame, color=[0,0,255], region=None):
        region_str = region.__str__()
        # Initialize prev data, if needed
        if not region_str in self.regions_frame_buffer:
            self.regions_frame_buffer[region_str] = FrameBuffer(self.buffer_size)
            self.regions_frame_buffer[region_str].fill(get_gray(frame, img_type='rgba'))
            # self.p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask = None, **self.feature_params)
            if region == None:
                self.regions_p0[region_str] = get_grid([0,0], self.default_size[0], self.default_size[1], self.num_points)
            else:
                self.regions_p0[region_str] = get_grid([region[0],region[1]], region[2], region[3], self.num_points)

        # Get appropriate data for the region
        p0 = self.regions_p0[region_str]
        prev_gray = self.regions_frame_buffer[region_str].pop()

        # Zero out the mask
        mask = np.zeros_like(frame)

        # Get gray images
        gray = get_gray(frame, img_type='rgba')


        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **self.lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]


        self.regions_frame_buffer[region_str].add_frame(gray)
        # self.p0 = good_new.reshape(-1,1,2)

        # Get static optical flow
        u = (good_new - good_old)/self.dt

        scale = 0.5/self.buffer_size
        for i,(pt,vec) in enumerate(zip(good_old,u)):
            a,b = pt.ravel()
            c,d = pt + scale*vec
            # c,d = old.ravel()
            mask = cv2.arrowedLine(mask, (a,b),(c,d), color, 1)
            frame = cv2.circle(frame,(a,b),2,color,-1)

        if not self.display_init:
            self.display = cv2.add(frame,mask)
            self.display_init = True
        else:
            self.display = cv2.add(self.display,mask)

        return u

    def display_image(self, win_name="Optical Flow"):
        self.display_init = False
        cv2.imshow(win_name,self.display)
        cv2.waitKey(1)
