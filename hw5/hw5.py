#!/usr/bin/env python
from my_cv.target_tracker import TargetTracker

import cv2
import numpy as np
import math

def onClick(event, x, y, flags, param):
    global tracker

    if event == cv2.EVENT_LBUTTONDOWN:
        tracker.reset()
        tracker.select_roi(frame)

def nothing():
    pass

cap = cv2.VideoCapture('mv2_001.avi')

cv2.namedWindow("Target Tracker")
cv2.namedWindow("HSV")
cv2.setMouseCallback("Target Tracker", onClick)

tracker = TargetTracker("CamShift")

# Get region of interest
ret, frame = cap.read()
tracker.select_roi(frame)

# HSV Testing
cv2.createTrackbar('hue min', 'HSV', 0, 255, nothing)
cv2.createTrackbar('hue max', 'HSV', 255, 255, nothing)
cv2.createTrackbar('sat min', 'HSV', 0, 255, nothing)
cv2.createTrackbar('sat max', 'HSV', 255, 255, nothing)
cv2.createTrackbar('val min', 'HSV', 0, 255, nothing)
cv2.createTrackbar('val max', 'HSV', 255, 255, nothing)
hue_min = 0
hue_max = 255
sat_min = 0
sat_max = 255
val_min = 0
val_max = 255

while True:
    ret, frame = cap.read()
    if ret == True:
        tracker.track_targets(frame)
        # hue_min = cv2.getTrackbarPos('hue min', 'HSV')
        # hue_max = cv2.getTrackbarPos('hue max', 'HSV')
        # sat_min = cv2.getTrackbarPos('sat min', 'HSV')
        # sat_max = cv2.getTrackbarPos('sat max', 'HSV')
        # val_min = cv2.getTrackbarPos('val min', 'HSV')
        # val_max = cv2.getTrackbarPos('val max', 'HSV')
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # hsv = cv2.inRange(hsv, (hue_min, sat_min, val_min), (hue_max, sat_max, val_max))
        #
        # cv2.imshow('hsv', hsv)
        cv2.waitKey(0)



    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('a'):
