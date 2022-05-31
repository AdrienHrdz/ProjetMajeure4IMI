import cv2
import cv2.aruco as aruco 
import numpy as np
import matplotlib.pyplot as plt

#aruco = cv2.aruco
p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
marker =  [0] * 4 #Initialisation

for i in range(len(marker)):
  marker[i] = aruco.drawMarker(p_dict, i, 75) # 75x75 px
  cv2.imwrite(f'./aruco/marker{i}.png', marker[i])