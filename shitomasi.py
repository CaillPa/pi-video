import numpy as np
import cv2

# Adresse de l'image en entrée
chemin = 'image/armoire.jpg'

# Parametres de la recherche
nbpoints = 12       # nombre d'angles a reporter
qualityLevel = 0.01 # seuil de qualité minimale pour les angles détectés
distancemin = 10    # distance minimale entre 2 angles a reporter

# Chargement de l'image et passage en NdG
image = cv2.imread(chemin)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Recherche des N angles les plus pertients, stockage dans corners
corners = cv2.goodFeaturesToTrack(gray, nbpoints, qualityLevel, distancemin)
corners = np.int0(corners)

# Placement des marqueurs sur l'image
for i in corners :
    x, y = i.ravel()
    cv2.circle(image, (x, y), 3, 255, -1)

# Affichage du resultat
cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.imshow('dst', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
