import numpy as np
import cv2

# Adresse de l'image en entrée
chemin = 'image/geo3.jpg'

# Chargement de l'image
img = cv2.imread(chemin, 0)

# Initialisaton du detecteur ORB
orb = cv2.ORB_create()

# Detection des angles avec ORB
kp = orb.detect(img, None)

# Calcule des trucs ?
kp, des = orb.compute(img, kp)

# Dessine les marqueurs sur l'image
img2 = cv2.drawKeypoints(img,kp,img,color=(0,255,0))

# Affichage du resultat
cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.imshow('dst', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
