import cv2
import numpy as np
import math

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

    def cnt(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []
