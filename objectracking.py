import numpy as np
from pivideostream import PiVideoStream
import cv2
import time

# Nom de la fenetre de prévisualisation
window_name = 'preview'
# Chemin de l'image a chercher dans le flux
img_path = 'image/cible.jpg'

# Creation du ORB detector et BF matcher
orb = cv2.create_ORB()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Chargement de l'image a chercher
img = cv2.imread(img_path, 0)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calcul des features et descriptors de l'image
kpimg, desimg = cv2.detectAndCompute(imgray, None)

# Creation de l'objet flux video + parametres optionels
vs = PiVideoStream()
vs.camera.video_stabilization = True
# Demarrage du flux video + temps de chauffe pour la camera
vs.start()
time.sleep(2.0)

# Creation de la fenetre d'affichage
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

while True :
    # Lecture d'une frame depuis le flux video
    frame = vs.read()

    # Recherche des points clé dans les 2 images
    if cv2.waitKey(1) & 0xFF == ord("q") :
        break

cv2.destroyAllWindows()
vs.stop()
