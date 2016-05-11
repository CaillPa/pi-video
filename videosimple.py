from pivideostream import PiVideoStream
from imutils.video import FPS # Pour les mesures de framerate
import cv2
import time

# Nom de la fenetre d'affichage
window_name = 'preview'

# Parametre du detecteur de Canny
thres1 = 100
thres2 = 200

# Taille du kernel pour le filtre Gaussien
gaussSize = (3, 3)

# Creation du thread de lecture + setup
vs=PiVideoStream()
vs.camera.video_stabilization = True
# Demarrage du flux video + warmup de la camera
vs.start()
time.sleep(1.0)

# Creation de la fenetre d'affichage
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

fps = FPS().start()

while True :
    frame = vs.read()

    # Passage en NdG + filtre Gaussien pour réduire le bruit
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, gaussSize, 0)
    
    # Application de l'algo de Canny
    edges = cv2.Canny(gray, thres1, thres2)

    imshow(window_name, edges)

    fps.update()

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") :
    	break

fps.stop()
print("Temps passé : {:.2f}".format(fps.elapsed()))
print("Approx. FPS : {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()