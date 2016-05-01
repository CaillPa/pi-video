from imutils.video.pivideostream import PiVideoStream
import cv2  # Librairie OpenCV
import time # Fonctions temporelles
import numpy as np

# Parametres de la recherche
nbpoints = 12       # nombre d'angles a reporter
qualityLevel = 0.01  # seuil de qualité minimale pour les angles détectés
distancemin = 2    # distance minimale entre 2 angles a reporter

# Creation de la fenetre d'affichage
cv2.namedWindow('angle', cv2.WINDOW_NORMAL)

# Créé un soustracteur d'arriere plan
fgbg = cv2.createBackgroundSubtractorMOG2(history = 200, detectShadows= False)

# Demarage du flux video sur un thread different
vs = PiVideoStream().start()
# Laisse le temps de chauffer a la camera
time.sleep(1.0)

# Boucle principale 
while True :
    # Recupere une frame et la passe en NdG
    frame = vs.read()
    
    # Applique le masque de soustraction
    fgmask = fgbg.apply(frame)
    
    # Effectue 2 passages d'erosion pour retirer les petits éléments
    fgmask = cv2.erode(fgmask, None, iterations = 2)  
    # Effectue 2 passages de dilatation pour conserver la taille
    fgmask = cv2.dilate(fgmask, None, iterations = 4)
    
    # Application du masque
    #res = cv2.bitwise_and(frame, frame, mask = fgmask)
    
    fgmask = np.uint8(fgmask)
    
    # Passage en NdG
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Recherche des N angles les plus pertients, stockage dans corners
    corners = cv2.goodFeaturesToTrack(gray, nbpoints, qualityLevel, distancemin, mask = fgmask)
    corners = np.int0(corners)
    
    # Placement des marqueurs sur l'image
    for i in corners :
        x, y = i.ravel()
        cv2.circle(frame, (x, y), 3, 255, -1)
        
    # Affichage du résultat
    cv2.imshow('angle', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Si on appuie sur q, arrete la boucle
    if key == ord("q") :
        break
        
# Ferme les fenetres ouvertes
cv2.destroyAllWindows()
vs.stop()
