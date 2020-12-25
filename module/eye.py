import cv2
import dlib

from imutils import face_utils

import numpy as np
import math

class Eye(object):
    LEFT_EYE_POINTS = [0, 1, 2, 3, 4, 5]
    RIGHT_EYE_POINTS = [6, 7, 8, 9, 10, 11]
    def __init__(self, landmarks, side, calibration):
        self.region = None
        self.calibration=calibration
        self.side=side
        self.ratio=None
        self.origin=None
        self._analyze(landmarks, side, calibration)

    def _analyze(self, landmarks, side, calibration):
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return
        self.region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        self.region = self.region.astype(np.int32)

        self.origin=(np.min(self.region[:, 0]), np.min(self.region[:, 1]))

        self.ratio = self.calc_wh_ratio(self.region)
        self.calibration.update(side, self.ratio)


    @staticmethod
    def _middle_point(p1, p2):
        x = int((p1[0] + p2[0]) / 2)
        y = int((p1[1] + p2[1]) / 2)
        return (x, y)

    def calc_wh_ratio(self, region):
        left = (region[0][0], region[0][1])
        right = (region[3][0], region[3][1])
        top = self._middle_point(region[1], region[2])
        bottom = self._middle_point(region[5], region[4])

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def calc_area(self, region):
        region=np.transpose(region)
        return 0.5*np.abs(np.dot(region[0],np.roll(region[1],1))-np.dot(region[1],np.roll(region[0],1)))

    def annotated_frame(self,frame):
        tmp = frame.copy()
        for(x,y) in self.region:
            cv2.circle(tmp, (x, y), 2, (255, 0, 255), -1)
        return tmp
        