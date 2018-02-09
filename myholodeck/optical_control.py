import sys
rm_python2 = []
for p in sys.path:
    if p.find('python2') != -1:
        rm_python2.append(p)
for p in rm_python2:
    sys.path.remove(p)

import numpy as np
import math
import cv2

from my_cv import *

class OpticalControl():
    def __init__(self):
        # Default params
        self.frame_width = 512
        self.frame_height = 512

        # CV class initialization
        op_flow_buffer_size = 3
        self.op_flow = OpticalFlow(op_flow_buffer_size)

        # Corridor following params
        self.corridor_color = [0,255,0]
        self.corridor_kp = 0.015/op_flow_buffer_size
        corridor_region_width = 150 # px
        corridor_region_height = 300
        corridor_region_y_offset = (self.frame_height - corridor_region_height)/2 # px
        self.corridor_region_left = [0, corridor_region_y_offset, corridor_region_width, corridor_region_height]
        self.corridor_region_right = [self.frame_width - corridor_region_width, corridor_region_y_offset,
                                                                    corridor_region_width, corridor_region_height]

    def follow_corridor(self, frame, display=True):
        # Get optical flow for the left and right regions
        left_flow = self.op_flow.compute_optical_flow(frame, self.corridor_color, self.corridor_region_left)
        right_flow = self.op_flow.compute_optical_flow(frame, self.corridor_color, self.corridor_region_right)

        # Get averaged x magnitudes
        # left_avg_norm = sum([u[0] for u in left_flow])/len(left_flow)
        # right_avg_norm = sum([u[0] for u in right_flow])/len(right_flow)
        left_avg_norm = sum([np.linalg.norm(u) for u in left_flow])/len(left_flow)
        right_avg_norm = sum([np.linalg.norm(u) for u in right_flow])/len(right_flow)

        vy_command = self.corridor_kp*(left_avg_norm - right_avg_norm)

        if display == True:
            self.op_flow.display_image('Follow corridor')

        return vy_command
