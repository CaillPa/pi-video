from pivideostream import PiVideoStream
import cv2  # Librairie OpenCV
import time # Fonctions temporelles
import serial

# Demarre la communication sur le port série
ser = serial.Serial('/dev/ttyACM0',9600)

# Fenetre d'affichage de la "vue" du robot
cv2.namedWindow('preview', cv2.WINDOW_NORMAL)

# Definition des seuils HSV pour le vert
greenLower = (90, 70, 90)
greenUpper = (105, 255, 255)

# Demarage du flux video sur un thread different
vs = PiVideoStream()
vs.start()
# Laisse le temps de chauffer a la camera
time.sleep(2)
vs.consistent()
print("Demarrage")

# Boucle principale 
while True :
    frame = vs.read()
    # Convertis la frame en HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Construis un masque pour la couleur définie
    mask = cv2.inRange(hsv, greenLower, greenUpper)

    mask = cv2.erode(mask, None, iterations = 4)  
    mask = cv2.dilate(mask, None, iterations = 4)
    # Detecte les contours dans le masque 
    # Calcul du moment du masque
    moment = cv2.moments(mask)
    try :
    # Recupere les coordonnées du centre du masque
        cx = int(moment['m10']/moment['m00'])
        cy = int(moment['m01']/moment['m00'])

    # Empeche la division par 0 lors du calcul du centre ..
    # .. si le masque est vide
    except ZeroDivisionError :
        cx = 0
        cy = 0

    frame = cv2.circle(frame, (cx, cy), 6, 255, -1)
    ser.write(str(cx).encode('ascii')+b'\r\n')
    recu = ser.readline()[:-2].decode('utf-8')
    print("recu : ", recu)

    # Affichage de la frame
    cv2.imshow('preview', frame)
    if cv2.waitKey(1) == ord("q") :
        break
# Ferme les fenetres ouvertes
cv2.destroyAllWindows()
vs.stop()
