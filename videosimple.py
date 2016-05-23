"""
    Lecture et affichage de la video
"""
from pivideostream import PiVideoStream
from imutils.video import FPS # Pour les mesures de framerate
import cv2
import time
import numpy as np

# Nom de la fenetre d'affichage
window_name = 'preview'

# Creation du thread de lecture + setup
vs=PiVideoStream()
vs.camera.video_stabilization = True
# Demarrage du flux video + warmup de la camera
vs.start()
time.sleep(2.0)

# Creation de la fenetre d'affichage
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

fps = FPS().start()

while True :

    frame = vs.read()
    fps.update()

    cv2.imshow(window_name, frame) 
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") :
    	break

fps.stop()
print("Temps passé : {:.2f}".format(fps.elapsed()))
print("Approx. FPS : {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
