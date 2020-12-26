import cv2
import dlib
from imutils import face_utils
import numpy as np

from .eye import *
from .calibration import *

class FaceTracker(object):
    predictor_path = './models/eye_predictor.dat'
    def __init__(self):
        self.frame=None
        self.face=None
        self.landmarks=None
        self.eye_left=None
        self.eye_right=None

        self.state=0

        self.calibration = Calibration() 
        self.face_detector = dlib.get_frontal_face_detector()
        self.eye_predictor = dlib.shape_predictor(self.predictor_path)

    def _analyze(self):
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(frame)
        try:
            self.face=faces[0]
            self.landmarks = self.eye_predictor(frame, faces[0])
            self.eye_left=Eye(self.landmarks,0,self.calibration)
            self.eye_right=Eye(self.landmarks,1,self.calibration)
            self.updateState()
        except IndexError:
            self.reset()

    def eye_blinking(self):
        if(self.face!=None):
            return self.eye_left.is_blinking() or self.eye_right.is_blinking()
        else:
            return False

    def eye_gazing(self):
        if(self.face!=None):
            return self.eye_left.is_gazing() or self.eye_right.is_gazing()
        else:
            return False

    def refresh(self, frame):
        self.frame = frame
        self._analyze()

    def reset(self):
        self.face=None
        self.landmarks=None
        self.eye_left=None
        self.eye_right=None
        self.calibration.reset()
    
    prev_blink=3
    def updateState(self):
        """
        State0: No Face, 
        State1: Calibration, 
        State2: Standard State,
        State3: Gazing,
        State4: Blinking
        """
        self.prev_blink+=1
        if(self.face==None):
            self.state=0
        elif(self.calibration.is_complete()):
            if(self.eye_blinking()):
                self.prev_blink=0
                self.state=4
            elif(self.eye_gazing() and self.prev_blink>9):
                self.state=3
            else:
                self.state=2
        else:
            self.state=1

    def annotated_frame(self):
        tmp = self.frame.copy()
        if(self.face!=None):
            #(x,y,w,h) = face_utils.rect_to_bb(self.face)
            #cv2.rectangle(tmp, (x, y), (x + w, y + h), (0, 255, 0), 2)

            landmarks = face_utils.shape_to_np(self.landmarks)
            for(x,y) in landmarks:
                cv2.circle(tmp, (x, y), 1, (0, 0, 255), -1)
            tmp=self.eye_left.annotated_frame(tmp)
            tmp=self.eye_right.annotated_frame(tmp)
            tmp=self.calibration.annotated_frame(tmp)

        return tmp
        