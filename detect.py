import cv2
from nishant.processing import ImageProcess
from nishant.gesture_detector import GestureDetector

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False

numFrame = 0
x, y, w, h = 0, 0, 300, 300
ip = ImageProcess()
gd = GestureDetector()
number = None
value = []

while ret:
    ret, frame = cap.read()
    cv2.rectangle(frame, (0, 0), (300, 300), (0, 255, 0), 1)
    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if numFrame<32:
        ip.update(blur)
    else:
        skin = ip.detect(blur)
        if skin is not None:
            thresh, cnts = skin
            #cv2.imshow('thresh', thresh)
            if cv2.contourArea(cnts)>1000:
                fingers = gd.detect(thresh,cnts,frame)
                #print('fingers:{}'.format(fingers))
                if number is None:
                    number = [1, fingers]
                else:
                    if number[1] == fingers:
                        number[0] +=1

                        if number[0] > 25:
                            number = None
                            if len(value) == 1:
                                value = []
                            value.append(fingers)
                    else:
                        number = None


    numFrame += 1

    if len(value)==1:
        GestureDetector.put_number(frame, value)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()

