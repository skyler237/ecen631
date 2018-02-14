import cv2
import numpy as np
import math

############ Main target tracker ############
class TargetTracker:
    def __init__(self, tracker_type="KLT"):
        tracker_types = {"KLT": KLTTracker,
                         "MeanShift": MeanShiftTracker,
                         "BGSubtraction": BGSubtractionTracker}
        self.tracker = tracker_types[tracker_type]
        print("Hello world!")

############ Individual tracker implementations ############
class Tracker:
    ''' Template class for tracking in an image
    '''
    def __init__(self):
        pass

class KLTTracker(Tracker):
    pass

class MeanShiftTracker(Tracker):
    pass

class BGSubtractionTracker(Tracker):
    pass
