# Detection et affichage des lignes doites dans un flux video
# Passage de la frame en NdG, filtre Gaussien
# Detecteur de contours de Canny puis HoughLines

from imutils.video.pivideostream import PiVideoStream
import cv2
import numpy as np
import time

# Parametre du detecteur de Canny
thres1 = 100
thres2 = 200

# Taille du kernel pour le filtre Gaussien
gaussSize = (3, 3)

# Nom de la fenetre d'affichage
window_name = 'Canny'

# Creation de la fenetre d'affichage
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Demarrage du flux de lecture
vs = PiVideoStream(framerate = 50).start()

# Timer pour laisser la camera se lancer
time.sleep(1.0)

# Boucle principale
while True :
    # Lecture de la frame dans le flux de lecture
    frame = vs.read()

    # Passage en NdG + filtre Gaussien pour réduire le bruit
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, gaussSize, 0)
    
    # Application de l'algo de Canny
    edges = cv2.Canny(gray, thres1, thres2)

    # Algo de Hough pour trouver les lignes
    lines = cv2.HoughLines(edges,1,np.pi/180,100)

    # Dessin des lignes sur la frame pour l'affichage
    if lines is not None :
	    for line in lines:
	    	for rho, theta in line :
	        	a = np.cos(theta)
	        	b = np.sin(theta)
	        	x0 = a*rho
	        	y0 = b*rho
	        	x1 = int(x0 + 1000*(-b))
	        	y1 = int(y0 + 1000*(a))
	        	x2 = int(x0 - 1000*(-b))
	        	y2 = int(y0 - 1000*(a))
	        	cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),1)

    # Affichage de la frame courante
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord("q") :
        break

# Ferme les fenetres ouvertes
cv2.destroyAllWindows()
vs.stop()
