from djitellopy import tello
import cv2
from cvzone.FaceDetectionModule import FaceDetector

import time

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamoff()
me.streamon()

detector = FaceDetector()
while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (640,480))
    img, bboxs = detector.findFaces(img, draw=True)
    cv2.imshow("Image", img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
