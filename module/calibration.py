import cv2
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt

class Calibration(object):
    def __init__(self):
        #for initial calibration
        self.num_init = 10
        self.initial_ratios_left = []
        self.initial_ratios_right = []

        #for realtime processing
        self.num_latest=3
        self.latest_ratios_left = [] 
        self.latest_ratios_right = []

        #thresholds
        self.thres_blink_left=None
        self.thres_blink_right=None
        self.thres_gaze_left=None
        self.thres_gaze_right=None

    def annotated_frame(self,frame):
        tmp=frame.copy()

        text="Blink: "+str(self.thres_blink_left)+" "+str(self.thres_blink_right)
        cv2.putText(tmp, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
        text="Gaze: "+str(self.thres_gaze_left)+" "+str(self.thres_gaze_right)
        cv2.putText(tmp, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
        return tmp

    #debug
    def show_graph_as_image(self):
        l=len(self.latest_ratios_left)
        if(l<self.num_latest):
            return 
        t=np.arange(1,l+1)
        fig, ax = plt.subplots()
        ax.plot(t, self.latest_ratios_left,label="left eye")
        ax.plot(t, self.latest_ratios_right, label="right eye")
        ax.set_title("Ratio between Eye Width and Height")
        ax.set_ylim(0,50)
        ax.grid(axis='both')
        ax.legend()
        fig.canvas.draw()
        im = np.array(fig.canvas.renderer.buffer_rgba())
        cv2.imshow("ratio",im)

        plt.close(fig)

    def sum_latest(self,side):
        try:
            if side == 0:
                return sum(self.latest_ratios_left)
            elif side == 1:
                return sum(self.latest_ratios_right)
        except Exception:
            return 0

    def max_latest(self,side):
        try:
            if side == 0:
                return max(self.latest_ratios_left)
            elif side == 1:
                return max(self.latest_ratios_right)
        except Exception:
            return 0

    #no realtime calibration for now
    def update_list(self,side,ratio):
        if(self.is_complete()):
            if(side==0):
                if(len(self.latest_ratios_left)<self.num_latest):
                    self.latest_ratios_left=self.initial_ratios_left[-1*self.num_latest:]
                else:
                    self.latest_ratios_left.pop(0)
                    self.latest_ratios_left.append(ratio)
            elif side == 1:
                if(len(self.latest_ratios_right)<self.num_latest):
                    self.latest_ratios_right=self.initial_ratios_right[-1*self.num_latest:]
                else:
                    self.latest_ratios_right.pop(0)
                    self.latest_ratios_right.append(ratio)

        else:
            if side == 0:
                if(len(self.initial_ratios_left)<=self.num_init):
                    self.initial_ratios_left.append(ratio)
            elif side == 1:
                if(len(self.initial_ratios_right)<=self.num_init):
                    self.initial_ratios_right.append(ratio)
            if(self.is_complete()):
                self.set_thres()
    
    def set_thres(self):
        min_left=min(self.initial_ratios_left)
        min_right=min(self.initial_ratios_right)
        self.thres_blink_left=min_left*8.5
        self.thres_blink_right=min_right*8.5
        self.thres_gaze_left=min_left*6.0
        self.thres_gaze_right=min_right*6.0

    def is_complete(self):
        return len(self.initial_ratios_left) >= self.num_init and len(self.initial_ratios_right) >= self.num_init

    def reset(self):
        self.ratios_left = []
        self.ratios_right = [] 
        