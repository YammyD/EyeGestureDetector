import cv2
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt

class Calibration(object):
    def __init__(self):
        self.nb_frames = 30

        self.ratios_left = [] 
        self.ratios_right = []

    def annotated_frame(self,frame):
        tmp=frame.copy()

        text="Left Ratio: "+str(int(self.calc_standard_ratio(0)))+", Right Ratio: "+str(int(self.calc_standard_ratio(1)))
        cv2.putText(tmp, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)

        return tmp

    #debug
    def show_graph_as_image(self):
        l=len(self.ratios_left)
        if(l==0):
            return 
        t=np.arange(1,l+1)
        fig, ax = plt.subplots()
        ax.plot(t, self.ratios_left,label="left eye")
        ax.plot(t, self.ratios_right, label="right eye")
        ax.set_title("Ratio between Eye Width and Height")
        ax.set_ylim(0,50)
        ax.grid(axis='both')
        ax.legend()
        fig.canvas.draw()
        im = np.array(fig.canvas.renderer.buffer_rgba())
        cv2.imshow("ratio",im)

        plt.close(fig)

    def update(self,side,ratio):
        if side == 0:
            if(len(self.ratios_left)<self.nb_frames):
                self.ratios_left.append(ratio)
            else:
                self.ratios_left.pop(0)
                self.ratios_left.append(ratio)
        elif side == 1:
            if(len(self.ratios_right)<self.nb_frames):
                self.ratios_right.append(ratio)
            else:
                self.ratios_right.pop(0)
                self.ratios_right.append(ratio)

    def calc_standard_ratio(self,side):
        try:
            if side == 0:
                return sum(self.ratios_left)/len(self.ratios_left)
            elif side == 1:
                return sum(self.ratios_right)/len(self.ratios_right)
        except ZeroDivisionError:
            return 5
        except TypeError:
            return 5

    def is_complete(self):
        return len(self.ratios_left) >= self.nb_frames and len(self.ratios_right) >= self.nb_frames

    def reset(self):
        self.ratios_left = []
        self.ratios_right = [] 
        