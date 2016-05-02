from imutils.video.pivideostream import PiVideoStream
import cv2  # Librairie OpenCV
import time # Fonctions temporelles
import numpy as np
import threading

# Parametres de la recherche
nbpoints = 12       # nombre d'angles a reporter
qualityLevel = 0.1  # seuil de qualité minimale pour les angles détectés
distancemin = 2    # distance minimale entre 2 angles a reporter

# Nom de la fenetre d'affichage
window_name = 'angle'

# Definition du Thread dédié a l'affichage
class threadAff(threading.Thread) :
    def __init__(self) :
        threading.Thread.__init__(self)
        self.frame = None
        self.isRunning = False
        self.key = None
        self.frameReady = False

    def run(self) :
        while self.isRunning :
            if self.frame is not None :
                if self.frameReady :
                    cv2.imshow(window_name, self.frame)
                    self.key = cv2.waitKey(1) & 0xFF
                    self.frameReady = False

    def updateFrame(self, frame) :
        self.frame = frame
        return self.key

# Creation de la fenetre d'affichage
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Creation du thread d'affichage
aff = threadAff()

# Créé un soustracteur d'arriere plan
fgbg = cv2.createBackgroundSubtractorMOG2(history = 200, detectShadows= False)

# Demarage du flux video sur un thread different
vs = PiVideoStream(framerate = 50).start()
# Laisse le temps de chauffer a la camera
time.sleep(1.0)

# Demmarage du thread d'affichage
aff.isRunning = True
aff.start()

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
    
    fgmask = np.uint8(fgmask)
    
    # Passage en NdG
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Recherche des N angles les plus pertients, stockage dans corners
    corners = cv2.goodFeaturesToTrack(gray, nbpoints, qualityLevel, distancemin, mask = fgmask)
    if corners is not None :
        corners = np.int0(corners)
    
        # Placement des marqueurs sur l'image
        for i in corners :
            x, y = i.ravel()
            cv2.circle(frame, (x, y), 2, 255, -1)
        
    # Affichage du résultat
    #cv2.imshow(window_name, frame)
    #cv2.waitKey(0)

    #print(threading.active_count())

    # Récupere l'eventuelle touche pressée et met a jour la frame courante
    if aff.updateFrame(frame) == ord("q") :
        aff.isRunning = False
        break
    
    # Indique au thread d'affichage qu'une nouvelle frame est prete
    aff.frameReady = True
        
# Ferme les fenetres ouvertes
cv2.destroyAllWindows()
vs.stop()
