import numpy as np
from pivideostream import PiVideoStream
import cv2
import time

# Nom de la fenetre de prévisualisation
window_name = 'preview'
# Nombre minimal de correspondances
min_match = 10

# Creation de l'objet flux video + parametres optionels
vs = PiVideoStream()
vs.camera.video_stabilization = True
# Demarrage du flux video + temps de chauffe pour la camera
vs.start()
time.sleep(2.0)

# Initialisation du SIFT
sift = cv2.xfeatures2d.SIFT_create()

# Creation de la fenetre d'affichage
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
