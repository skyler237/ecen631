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
        self.f = self.frame_width/2

        # Corridor following params
        self.corridor_color = [0,255,0]
        self.corridor_kp = 0.01/op_flow_buffer_size
        corridor_region_width = 160 # px
        corridor_region_height = 250
        corridor_region_y_offset = (self.frame_height - corridor_region_height)/2 # px
        self.corridor_region_left = [0, corridor_region_y_offset, corridor_region_width, corridor_region_height]
        self.corridor_region_right = [self.frame_width - corridor_region_width, corridor_region_y_offset,
                                                                    corridor_region_width, corridor_region_height]


        # Obstacle avoidance params
        self.obs_color = [0,0,255]
        self.obs_kp = 0.0025/op_flow_buffer_size
        obs_region_x_offset = 80
        obs_region_width = self.frame_width/2 - obs_region_x_offset # px
        obs_region_height = 220
        obs_region_y_offset = 80 # px
        self.obs_region_left = [obs_region_x_offset, obs_region_y_offset, obs_region_width, obs_region_height]
        self.obs_region_right = [self.frame_width - (obs_region_width + obs_region_x_offset), obs_region_y_offset,
                                                                    obs_region_width, obs_region_height]


        # Altitude hold params
        self.altitude_color = [255,0,0]
        self.altitude_kp = 0.003/op_flow_buffer_size
        self.max_hdot = 1.0 # m/s
        self.alt_vel_ratio = 1.0
        alt_region_height = 180
        alt_region_width = 300
        alt_region_x_offset = (self.frame_width - alt_region_width)/2
        alt_region_y_offset = self.frame_height - alt_region_height
        self.alt_region = [alt_region_x_offset, alt_region_y_offset, alt_region_width, alt_region_height]


        # Collision detection
        self.collision_color = [255,255,0]



    def follow_corridor(self, frame, display=True, ang_vel=[0., 0., 0.]):
        # Get optical flow for the left and right regions
        left_flow = self.op_flow.compute_optical_flow(frame, self.corridor_color, self.corridor_region_left,
                                                        ang_vel=ang_vel, average=True)
        right_flow = self.op_flow.compute_optical_flow(frame, self.corridor_color, self.corridor_region_right,
                                                        ang_vel=ang_vel, average=True)

        # Get averaged x magnitudes
        # left_avg_norm = sum([u[0] for u in left_flow])/len(left_flow)
        # right_avg_norm = sum([u[0] for u in right_flow])/len(right_flow)
        # left_avg_norm = sum([np.linalg.norm(u) for u in left_flow])/len(left_flow)
        # right_avg_norm = sum([np.linalg.norm(u) for u in right_flow])/len(right_flow)
        left_avg_norm = np.linalg.norm(left_flow)
        right_avg_norm = np.linalg.norm(right_flow)


        vy_command = self.corridor_kp*(left_avg_norm - right_avg_norm)

        if display == True:
            self.op_flow.display_image()

        return vy_command

    def avoid_obstacles(self, frame, yaw, dt, display=True, ang_vel=[0., 0., 0.]):
        # Get optical flow for the left and right regions
        left_flow = self.op_flow.compute_optical_flow(frame, self.obs_color, self.obs_region_left,
                                                        ang_vel=ang_vel, average=True)
        right_flow = self.op_flow.compute_optical_flow(frame, self.obs_color, self.obs_region_right,
                                                        ang_vel=ang_vel, average=True)
        left_avg_norm = left_flow[0]
        right_avg_norm = right_flow[0]

        r_command = self.obs_kp*(left_avg_norm+right_avg_norm)
        yaw_command = yaw + r_command*dt
        if yaw_command > math.pi:
            yaw_command -= 2*math.pi
        elif yaw_command < -math.pi:
            yaw_command += 2*math.pi

        if display == True:
            self.op_flow.display_image()

        return yaw_command

    def altitude_hold(self, frame, altitude, dt, display=True, ang_vel=[0., 0., 0.]):
        alt_flow,points = self.op_flow.compute_optical_flow(frame, self.altitude_color, self.alt_region, ang_vel=ang_vel, average=False)

        alt_vel_estimates = np.zeros((np.shape(alt_flow)[0], 1))
        for i,(u,p) in enumerate(zip(alt_flow,points)):
            if u[1] != 0:
                sy = p[1] - self.frame_height/2
                sydot = u[1]
                alt_vel_estimates[i] = (self.f/np.sqrt(sy**2 + self.f**2))*(sy/sydot)

        # alt_vel_estimates = [(self.f/np.sqrt(p[1]**2 + self.f**2))*(p[1]/u[1]) for (u,p) in zip(alt_flow,points)]
        avg_alt_vel_estimate = np.average(alt_vel_estimates)

        hdot = max(min(self.alt_vel_ratio - avg_alt_vel_estimate, self.max_hdot), -self.max_hdot)

        alt_command = altitude + hdot*dt

        if display == True:
            self.op_flow.display_image()

        return alt_command
