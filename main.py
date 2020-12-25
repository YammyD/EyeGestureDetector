import cv2
import dlib

import numpy as np

from module import *

if __name__=="__main__":
    #execute v4l2-ctl -d /dev/video0 --list-formats-ext
    #for possible resolution and fps
    DEVICE_ID = 0
    WIDTH = 800
    HEIGHT = 600
    FPS=30

    #Capture
    cap=cv2.VideoCapture(DEVICE_ID)
    assert cap.isOpened(), 'Could not open video device'
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))

    STATE=["No Face Found","Calibrating","Normal", "Gazing", "Blinking"]

    tracker=FaceTracker()
    while(True):
        ret, frame = cap.read()

        tracker.refresh(frame)
        frame=tracker.annotated_frame()

        #debug
        #tracker.calibration.show_graph_as_image()

        state=tracker.getState()
        text=STATE[state]
        cv2.putText(frame, text,(30,HEIGHT-30),cv2.FONT_HERSHEY_DUPLEX, 2.0, (147, 58, 31), 2)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
