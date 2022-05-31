import cv2
import numpy as np
import matplotlib.pyplot as plt
from divers import LECTURE_IMAGE

filename = './aruco./IMG_20220531_144812.jpg'

IMAGE = LECTURE_IMAGE(filename)
plt.figure()
plt.subplot(121)
plt.imshow(cv2.cvtColor(IMAGE,cv2.COLOR_BGR2RGB))

ret,thresh1=cv2.threshold(IMAGE,125,255,cv2.THRESH_BINARY_INV)
element_structurant = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
IMAGE = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, element_structurant)
plt.subplot(122)
plt.imshow(cv2.cvtColor(IMAGE,cv2.COLOR_BGR2RGB))
plt.show()
