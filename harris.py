import numpy as np
import cv2

# Adresse de l'image en entrée
chemin = 'image/geo3.jpg'

# Parametres de la fonction cornerHarris()
bSize = 2       # Taille du voisinage pour la detection d'angle
kSize = 11      # Parametre d'ouverture du filtre de Sobel
kHarris = 0.04  # Coef de l'equation de Harris

# Paramatre de seuil pour le résultat
threshold = 0.01

# Chargement de l'image et passage en NdG
image = cv2.imread(chemin)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# L'image est convertie en float32 pour satisfaire cornerHarris()
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, bSize, kSize, kHarris)

# Augmente la taille des marqueurs
#dst = cv2.dilate(dst, None)

# Seuillage et placement des points
image[dst>threshold*dst.max()] = [0, 0, 255]

# Affichage du resultat
cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.imshow('dst', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite('corner23004001.png', image)
