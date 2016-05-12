import numpy as np
from pivideostream import PiVideoStream
import cv2
import time

# Nom de la fenetre de prévisualisation
window_name = 'preview'
# Chemin de l'image a chercher dans le flux
img_path = 'image/patafix.jpg'

MIN_MATCH_COUNT = 10

# Creation du ORB detector et BF matcher
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)

# Chargement de l'image a chercher
img = cv2.imread(img_path, 0)

# Calcul des features et descriptors de l'image
kpimg, desimg = orb.detectAndCompute(img, None)

# Creation de l'objet flux video + parametres optionels
vs = PiVideoStream()
vs.camera.video_stabilization = True
# Demarrage du flux video + temps de chauffe pour la camera
vs.start()
time.sleep(2.0)

# Creation de la fenetre d'affichage
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

while True :
    # Lecture d'une frame depuis le flux video + passage NdG
    frame = vs.read()

    # Calcul des features et des descriptors de l'image
    kpframe, desframe = orb.detectAndCompute(frame, None)

    # Calcul des correspondances
    matches = bf.knnMatch(desimg, desframe, k = 2)

    # Tri des correspondances
    good = []
    for m, n in matches :
        if m.distance < 0.75 * n.distance :
            good.append(m)
    #matches = sorted(matches, key = lambda x:x.distance)

    # Dessin des matches sur l'image
    #res = cv2.drawMatchesKnn(frame, kpframe, img, kpimg, good, None, flags = 2)

    if len(good)>MIN_MATCH_COUNT:
        print("Enough matches")
        src_pts = np.float32([ kpimg[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpframe[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        frame = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print ("Not enough matches are found")
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)

    img3 = cv2.drawMatches(img,kpimg,frame,kpframe,good,None,**draw_params)

    good.clear()
    cv2.imshow(window_name, img3)
    if cv2.waitKey(1) & 0xFF == ord("q") :
        break

cv2.destroyAllWindows()
vs.stop()
