import numpy as np
from pivideostream import PiVideoStream
import cv2
import time
from imutils.video import FPS

window_name = 'preview'

vs = PiVideoStream()
vs.start()
time.sleep(2.0)

cv2.namedWindow(window_name, cv2.WINDOW_OPENGL)

fps = FPS().start()
while True :
    frame = vs.read()

    fps.update()

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord("q") :
        break

fps.stop()

print("Temps passé : {:.2f}".format(fps.elapsed()))
print("Approx. FPS : {:.2f}".format(fps.fps()))
vs.stop()
cv2.destroyAllWindows()
