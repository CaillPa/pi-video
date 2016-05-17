import numpy as np
from pivideostream import PiVideoStream
import cv2
import time
from imutils.video import FPS # Pour les mesures de framerate

# Nom de la fenetre de prévisualisation
window_name = 'preview'

# Creation de l'objet flux video + parametres optionels
vs = PiVideoStream()
vs.camera.video_stabilization = True
# Demarrage du flux video + temps de chauffe pour la camera
vs.start()
time.sleep(2.0)

# Creation de la fenetre d'affichage
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

# Position initiale de la fenetre
r,h,c,w = 10,120,10,100  # valeurs codées en dur
track_window = (c,r,w,h)

# Premiere boucle pour placer l'objet a traquer
while True :
	frame = vs.read()

	# Dessine le rectangle de selection
	frame = cv2.rectangle(frame, (r, c), (r+h, c+w), (255, 0, 0), thickness = 2)

	cv2.imshow(window_name,frame)
	if cv2.waitKey(1) & 0xFF == ord("q") :
		break

# setup de la region de tracking
roi = frame[r:r+h, c:c+w]
# Conversion en HSV
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# Definis un masque pour filtrer les couleurs inutiles
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# Calcul de l'histo normalisé
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Conditions d'arret
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

fps = FPS().start()

while True:
	frame = vs.read()
    # Passage en HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

	# Applique le camshift
	ret, track_window = cv2.CamShift(dst, track_window, term_crit)

	# Draw it on image
	pts = cv2.boxPoints(ret)
	pts = np.int0(pts)
	img2 = cv2.polylines(frame,[pts],True, 255,2)

	fps.update()

	cv2.imshow(window_name,img2)
	if cv2.waitKey(1) & 0xFF == ord("q") :
		break

fps.stop()
print("Temps passé : {:.2f}".format(fps.elapsed()))
print("Approx. FPS : {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
