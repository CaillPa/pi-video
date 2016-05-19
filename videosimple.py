from pivideostream import PiVideoStream
from imutils.video import FPS # Pour les mesures de framerate
import cv2
import time
import numpy as np

# Nom de la fenetre d'affichage
window_name = 'preview'

# Taille du kernel pour le filtre Gaussien
gaussSize = (3, 3)

ddepth = cv2.CV_16S

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

    # Passage en NdG + filtre Gaussien pour réduire le bruit
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, gaussSize, 1)

    """
    OPTI MULTITHREAD
    """
    # Gradient-X
    grad_x = cv2.Sobel(gray,ddepth,1,0,ksize=3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
    # Gradient-Y
    grad_y = cv2.Sobel(gray,ddepth,0,1,ksize=3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    dst = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
    ret,dst = cv2.threshold(dst,50,255,cv2.THRESH_BINARY)

    fps.update()

    cv2.imshow(window_name, dst) 
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") :
    	break

fps.stop()
print("Temps passé : {:.2f}".format(fps.elapsed()))
print("Approx. FPS : {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
