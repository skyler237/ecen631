import cv2
import numpy as np
import math
import yaml

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

def get_random_color():
    return np.random.randint(0,255,(1,3))

def draw_features(img, features, size=2, color=(0,0,255)):
    for i,(feat) in enumerate(features):
        a,b = feat.ravel()
        img = cv2.circle(img,(int(a),int(b)),size,color,-1)
    return img

def display_features(frame, features):
    frame = np.copy(frame)
    frame = draw_features(frame, features)
    cv2.imshow('Features', frame)
    cv2.waitKey(1)

class CamParams:
    def __init__(self, param_file):
        self.params = yaml.load(open(param_file))
        self.K = self.get_K()
        self.width = self.params['image_width']
        self.height = self.params['image_height']

    def get_K(self):
        camera_matrix = self.params['camera_matrix']
        K = np.reshape(camera_matrix['data'], (camera_matrix['rows'], camera_matrix['cols']))
        return K


class FrameBuffer:
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

    def get_frames(self):
        return self.buffer

    def cnt(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []

class BackgroundSubtractor:
    def __init__(self, display=False):
        # Class option
        self.display = display

        # Initialize backgroud subtractor
        self.bgsub = cv2.createBackgroundSubtractorMOG2()

        # Set parameters
        self.bgsub.setHistory(10)                        # default = 500
        self.bgsub.setNMixtures(1)                       # default = 5
        self.bgsub.setDetectShadows(True)               # default = True
        self.bgsub.setBackgroundRatio(0.9)               # default = 0.9
        self.bgsub.setVarThresholdGen(16.0)              # defualt = 9.0
        self.bgsub.setVarThreshold(40.0)                 # defualt = 16.0
        self.bgsub.setComplexityReductionThreshold(0.05) # default = 0.05
        self.learning_rate = -1                          # default = -1

        self.open_kernel = np.ones((2,2),np.uint8)
        self.close_kernel = np.ones((8,8),np.uint8)

        # Initialize variables
        self.prev_frame = None

    def get_fg(self, frame):
        fg = self.get_fg_raw(frame)
        if len(self.open_kernel) > 0:
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.open_kernel)
        if len(self.close_kernel) > 0:
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.close_kernel)
        if self.display:
            cv2.imshow("Backround Subtraction", fg)
            # cv2.waitKey(1)
        return fg

    def get_fg_raw(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame
        # fg = self.bgsub.apply(frame, learningRate=self.learning_rate)
        fg = self.bgsub.apply(frame)
        return fg
