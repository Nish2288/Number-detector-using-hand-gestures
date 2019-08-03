import cv2
from sklearn.metrics import pairwise
import numpy as np
import imutils
class GestureDetector:
    def __init__(self):
        pass

    def detect(self, thresh, cnts, frame):
        hull = cv2.convexHull(cnts)

        # Find extreme point in hull
        extLeft  = tuple(hull[hull[:, :, 0].argmin()][0])
        extRight = tuple(hull[hull[:, :, 0].argmax()][0])
        extTop = tuple(hull[hull[:, :, 1].argmin()][0])
        extBot = tuple(hull[hull[:, :, 1].argmax()][0])
        cv2.drawContours(frame, [hull], -1, (0,255,0), 1)
        cv2.circle(frame, extLeft, 8, (255,0,0), -1)
        cv2.circle(frame, extRight, 8, (255,0,0), -1)
        cv2.circle(frame, extTop, 8, (255,0,0), -1)
        cv2.circle(frame, extBot, 8, (255,0,0), -1)

        # Find center of hull
        cX = (extLeft[0] + extRight[0])//2
        cY = (extTop[1] + extBot[1])//2
        cY += int(0.15*cY) 
        cv2.circle(frame, (cX, cY), 8, (255,0,0), -1)

        # Find max distance from center
        D = pairwise.euclidean_distances([(cX, cY)], [extLeft, extRight, extTop, extBot])[0]
        D = D[D.argmax()]
        
        # We take radius as 60% of max distance
        r = int(0.64 *D)
        circum = 2 * np.pi * r
        
        cv2.circle(frame, (cX, cY), r, (0,255,0), 1)

        circleROI = np.zeros(thresh.shape[0:2], dtype='uint8')
        cv2.circle(circleROI, (cX, cY), r, 255, 1)
        circleROI = cv2.bitwise_and(thresh, thresh, mask=circleROI)
        cv2.imshow('ROI', circleROI)
        cnts = cv2.findContours(circleROI.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        
        total = 0
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if c.shape[0]< circum * 0.25 and c.shape[0]>0.03 * circum and (y+h) < (cY + 0.25*cY):
                total += 1

        return total

    @staticmethod
    def put_number(frame, num):
        cv2.putText(frame, "Prediction :  " + str(num[0]), (310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)