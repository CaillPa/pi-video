from imutils.video.pivideostream import PiVideoStream
import cv2
import numpy as np
import time

# Parametre du detecteur de Canny
thres1 = 100
thres2 = 200
ouverture = 3

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

    # Passage en NdG
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # Application de l'algo de Canny
    edges = cv2.Canny(gray, thres1, thres2, apertureSize = ouverture)

    lines = cv2.HoughLines2(edges,1,np.pi/180,10)

    #if lines is not None :
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
            
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

    # Affichage de la frame courante
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord("q") :
        break

# Ferme les fenetres ouvertes
cv2.destroyAllWindows()
vs.stop()
