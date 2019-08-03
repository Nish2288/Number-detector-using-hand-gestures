import cv2
import imutils

class ImageProcess:
    # Constructor with Accumulated weights.
    # Heigher the weight more current frame contribution
    def __init__(self, accuWeight = 0.5):
        self.accuWeight = accuWeight
        # Initialize background model
        self.bg = None

    def update(self, image):
        if self.bg is None:
            self.bg = image.copy().astype('float')
            return
        # accumulate weighted average between background model and current image 
        # and store in self.bg
        
        cv2.accumulateWeighted(image, self.bg, self.accuWeight)
        
    def detect(self, image):
        # Calculate absolute difference between background model and image
        
        delta = cv2.absdiff(self.bg.astype('uint8'), image)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) == 0:
            return None
        cv2.imshow('thresh', thresh)
        #print('Thresh2')
        return(thresh, max(cnts, key=cv2.contourArea))