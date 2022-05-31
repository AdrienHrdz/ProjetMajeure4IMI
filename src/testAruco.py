import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt

p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
img = cv2.imread('./aruco./IMG_20220531_144812.jpg')
corners, ids, rejectedImgPoints = aruco.detectMarkers(img, p_dict) #détection

#Stockez les "coordonnées centrales" du marqueur en m dans l'ordre à partir du coin supérieur gauche dans le sens des aiguilles d'une montre
m = np.empty((4,2))
for i,c in zip(ids.ravel(), corners):
  m[i] = c[0].mean(axis=0)

width, height = (500,500) #Taille de l'image après transformation

marker_coordinates = np.float32(m)
true_coordinates   = np.float32([[0,0],[width,0],[width,height],[0,height]])
trans_mat = cv2.getPerspectiveTransform(marker_coordinates,true_coordinates)
img_trans = cv2.warpPerspective(img,trans_mat,(width, height))

cv2.imwrite('test.jpg', img_trans)
plt.imshow(img_trans, 'gray')
plt.show()