from imutils.video.pivideostream import PiVideoStream
import cv2  # Librairie OpenCV
import time # Fonctions temporelles

# Definition des seuils HSV pour le vert
greenLower = (60, 100, 70)
greenUpper = (95, 255, 130)

# Creation de la fenetre d'affichage
cv2.namedWindow('opti', cv2.WINDOW_NORMAL)

# Demarage du flux video sur un thread different
vs = PiVideoStream().start()

# Laisse le temps de chauffer a la camera
time.sleep(1.0)

# Boucle principale 
while True :
    
    frame = vs.read()
    #frame = imutils.resize(frame, width = 400)
    
    # Convertis la frame en HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Construis un masque pour la couleur définie
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    
    # Effectue 2 passages d'erosion pour retirer les petits éléments
    mask = cv2.erode(mask, None, iterations = 2)  
    # Effectue 2 passages de dilatation pour conserver la taille
    mask = cv2.dilate(mask, None, iterations = 2)
    
    # Detecte les contours dans le masque 
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    # Si on a trouvé un contours
    if len(cnts) > 0 :
        # Dans c on met le plus grand contours trouvé
        c = max(cnts, key = cv2.contourArea)
        
        # On utlise c pour calculer le cercle qui l'entours
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        
        # Si le cercle a un rayon minimum
        if radius > 6 :
            # Dessine le cercle sur la frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0,255, 255), 2)
    
    # Affiche l'image a l'ecran
    cv2.imshow('opti', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Si on appuie sur q, arrete la boucle
    if key == ord("q") :
        break
        
# Ferme les fenetres ouvertes
cv2.destroyAllWindows()
vs.stop()
