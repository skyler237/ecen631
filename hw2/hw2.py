import cv2
import numpy as np

NORMAL = 0
BLUR = 1
EDGE = 2

output_rows = 1
output_cols = 2

cam = cv2.VideoCapture(0)

num_filters = 3

blur_on = True
edge_on = False
color_select_on = True
mode = NORMAL
blur_val = 5
edge_min = 100
edge_max = 200
hue_min = 0
hue_max = 255

def select_hue(img, hue_min, hue_max):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    # ret, h = cv2.threshold(h, hue_min, 255, cv2.THRESH_TOZERO)
    # cv2.imshow('h1', h)
    # ret, h = cv2.threshold(h, hue_max, 255, cv2.THRESH_TOZERO_INV)
    mask = cv2.inRange(h, hue_min, hue_max)
    # mask = cv2.GaussianBlur(mask, (3,3), 0)
    # mask = cv2.fastNlMeansDenoising(mask,None,10,7,21)
    mask = cv2.medianBlur(mask, 5)
    # mask = cv2.bilateralFilter(mask,9,75,75)
    # cv2.imshow('mask', mask)
    s = cv2.bitwise_and(s,s,mask=mask)
    hsv = cv2.merge((h,s,v))

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def nothing(x):
    pass

cv2.namedWindow('output')
cv2.createTrackbar('blur', 'output', 0, 100, nothing)
cv2.createTrackbar('edge min', 'output', 0, 500, nothing)
cv2.createTrackbar('edge max', 'output', 1, 500, nothing)
cv2.createTrackbar('hue min', 'output', 0, 255, nothing)
cv2.createTrackbar('hue max', 'output', 255, 255, nothing)


while True:
    # Get the frame
    ret_val, img = cam.read()

    rows = np.size(img,0)
    cols = np.size(img,1)
    chnls = np.size(img,2)
    output = np.zeros((output_rows*rows, output_cols*cols, chnls))
    row1_slice = slice(0,rows)
    col1_slice = slice(0,cols)
    col2_slice = slice(cols,2*cols)

    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check for mode change
    key = cv2.waitKey(1)
    if key == ord('b'):
        print("Blur mode")
        blur_on = blur_on ^ True # Toggle
    elif key == ord('e'):
        print("Edge mode")
        edge_on = edge_on ^ True # Toggle
    elif key == ord('n'):
        print("Normal mode")
        blur_on = False
        edge_on = False
        mode = NORMAL
    elif key == 27:
        break # Exit on 'ESC' press

    # grab values from trackbars
    blur_val = (cv2.getTrackbarPos('blur', 'output'))*2 + 1
    edge_min = cv2.getTrackbarPos('edge min', 'output')
    edge_max = cv2.getTrackbarPos('edge max', 'output')
    hue_min = cv2.getTrackbarPos('hue min', 'output')
    hue_max = cv2.getTrackbarPos('hue max', 'output')
    if edge_min == 0 and edge_max == 0:
        edge_on = False
    else:
        edge_on = True

    # if mode == NORMAL:
    #     continue # just leave the image alone
    # Perform operations
    if blur_on:
        # perform blur operation on image
        img = cv2.GaussianBlur(img, (blur_val,blur_val), 0)
        gray = cv2.GaussianBlur(gray, (blur_val,blur_val), 0)
    if edge_on:
        # Perform canny edge detection
        img = cv2.Canny(img, edge_min, edge_max)
        gray = cv2.Canny(gray, edge_min, edge_max)
    if color_select_on and not edge_on:
        img = select_hue(img, hue_min, hue_max)

    # Display image
    if np.size(np.shape(img)) > 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    output = np.hstack((img,gray))
    cv2.imshow('output', output)

cv2.destroyAllWindows()
