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

cap = cv2.VideoCapture('mv2_001.avi')

cv2.namedWindow("Target Tracker")
cv2.setMouseCallback("Target Tracker", onClick)

tracker = TargetTracker("KLT")

# Get region of interest
ret, frame = cap.read()
tracker.select_roi(frame)

while True:
    ret, frame = cap.read()
    if ret == True:
        tracker.track_targets(frame)

    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('a'):
