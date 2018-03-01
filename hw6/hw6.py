#!/usr/bin/env python
import cv2
import numpy as np
import math

def onClick(event, x, y, flags, param):
    global frame, frame_prev
    if event == cv2.EVENT_LBUTTONDOWN:
        # Process frame to get R,T difference
        

        # Update previous frame
        frame_prev = frame

def nothing():
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Webcam", onClick)

# Get region of interest
ret, frame_prev = cap.read()

while True:
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow("Webcam", frame)
        cv2.imshow("Prev Frame", frame_prev)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
