import numpy as np
import cv2

class MultiImage():
    def __init__(self, rows=1, cols=1):
        self.rows = rows
        self.cols = cols

        self.images = np.zeros((rows,cols))
        self.image_size = [0,0,0]

    def add_image(self, img, row, col):
        img = self.get_3ch_img(img)
        self.images[row,col] = img
        if self.image_size == [0,0]:
            self.image_size = np.shape(img)

    def get_3ch_image(self,img):
        channels = np.shape(img)[2]

        if channels == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif channels == 3:
            return img
        elif channels == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            print("Invalid number of channels: {0}".format(channels))

    def get_display(self):
        display = []
        for i in range(0,self.rows):
            row = []
            for j in range(0,self.cols):
                # Stack images across
                img = self.images[i,j]
                # Blank image if we haven't added one yet
                if np.shape(img) != self.image_size:
                    img = np.zeros(self.image_size)
                if j == 0:
                    row = img
                else:
                    row = np.hstack((row,img))
            # Stack rows down
            if i == 0:
                display = row
            else:
                display = np.vstack((display,row))
        return display
