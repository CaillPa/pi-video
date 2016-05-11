import numpy as np
from pivideostream import PiVideoStream
import cv2
import time

# Nom de la fenetre de prévisualisation
window_name = 'preview'
# Chemin de l'image a chercher dans le flux
img_path = 'image/cible.jpg'
# Nombre minimal de correspondances
min_match = 10
# Parametres du FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

# Parametres de dessin des résultats
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
singlePointColor = None,
matchesMask = matchesMask, # draw only inliers
flags = 2)

# Initialisation du SIFT
sift = cv2.xfeatures2d.SIFT_create()

# Initialisation du FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Chargement de l'image a chercher
img = cv2.imread(img_path, 0)

# Creation de l'objet flux video + parametres optionels
vs = PiVideoStream()
vs.camera.video_stabilization = True
# Demarrage du flux video + temps de chauffe pour la camera
vs.start()
time.sleep(2.0)

# Creation de la fenetre d'affichage
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

while True :
    # Lecture d'une frame depuis le flux video
    frame = vs.read()

    # Recherche des points clé dans les 2 images
    kpimg, desimg = sift.detectAndCompute(img, None)
    kpframe, desframe = sift.detectAndCompute(frame, None)

    matches = flann.knnMatch(desimg, desframe, k = 2)

    # Stocke les bons matchs
    good = []
    for m, n in matches :
        if m.distance < 0.7 * n.distance :
            good.append(m)

    # Teste si on a assez de bonnes correspondances
    if len(good) < min_match :
        print("Pas assez de correspondances !")
        matchesMask = None

    else :
        src_pts = np.float32([kpimg[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpframe[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w, d = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        
    img3 = cv2.drawMatches(img1, kpimg, img2, kp2, good, None, **draw_params)

    cv2.imshow(window_name, img3)

    if cv2.waitKey(1) & 0xFF == ord("q") :
        break

cv2.destroyAllWindows()
vs.stop()
