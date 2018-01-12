import cv2

NORMAL = 0
BLUR = 1
EDGE = 2

cam = cv2.VideoCapture(0)

blur_on = False
edge_on = False
mode = NORMAL
blur_val = 5
edge_min = 100
edge_max = 200

while True:
    # Get the frame
    ret_val, img = cam.read()

    img = cv2.flip(img, 1)

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


    # if mode == NORMAL:
    #     continue # just leave the image alone
    # Perform operations
    if edge_on:
        if key == ord('i'):
            edge_max += 10
            print("Edge max increased to {0}".format(edge_max))
        elif key == ord('k'):
            edge_max -= 10
            if edge_max < 10: # limit how far this can go
                edge_max = 10
            print("Edge max decreased to {0}".format(edge_max))
        if key == ord('o'):
            edge_min += 10
            print("Edge min increased to {0}".format(edge_min))
        elif key == ord('l'):
            edge_min -= 10
            if edge_min < 0: # limit how far this can go
                edge_min = 0
            print("Edge min decreased to {0}".format(edge_min))
        # perform edge detection on image
        img = cv2.Canny(img, edge_min, edge_max)
    if blur_on:
        # Adjust params if key is pressed
        if key == ord('u'):
            blur_val += 2
            print("Blur val increased to {0}".format(blur_val))
        elif key == ord('j'):
            blur_val -= 2
            if blur_val < 1: # limit how far this can go
                blur_val = 1
            print("Blur val decreased to {0}".format(blur_val))
        # perform blur operation on image
        img = cv2.GaussianBlur(img, (blur_val,blur_val), 0)

    # Display image
    cv2.imshow('my webcam', img)

cv2.destroyAllWindows()
