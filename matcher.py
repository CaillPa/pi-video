import numpy as np
import cv2
import matplotlib.pyplot as plt

# Adresses des images en entrée
isolee = 'image/clerouge.jpg'
image = 'image/clerougedecors.jpg'

# Chargement des images en mémoire
img1 = cv2.imread(isolee, 0)
img2 = cv2.imread(image, 0)

# Inlitialisation du detecteur ORB
orb = cv2.ORB_create()

# Recherche des points-cle avec l'ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Creation d'un objet Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1, des2)

# Classe les correspondances en fonction de leur distance
matches = sorted(matches, key = lambda x:x.distance)

# Dessine les 10 premieres corrspondances
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags = 2)

plt.imshow(img3), plt.show()
