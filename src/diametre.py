import cv2
import numpy as np
import matplotlib.pyplot as plt
from divers import LECTURE_IMAGE

filename = './aruco./IMG_20220531_170740.jpg'

IMAGE = LECTURE_IMAGE(filename)
IMAGE = cv2.cvtColor(IMAGE, cv2.COLOR_RGB2GRAY)

ret,thresh1=cv2.threshold(IMAGE,125,255,cv2.THRESH_BINARY_INV)
bweuler, array_components = cv2.connectedComponents(IMAGE.astype(np.uint8))
print(bweuler)
plt.imshow(thresh1, cmap='gray')
plt.show()

n = 0
bweuler01, bweuler02 = 0, 0
while bweuler01 == bweuler02:
    n += 10
    bweuler01 = cv2.connectedComponents(IMAGE.astype(np.uint8))[0]
    element_structurant = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
    IMAGE_OPEN = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, element_structurant)
    bweuler02 = cv2.connectedComponents(IMAGE_OPEN.astype(np.uint8))[0]
    print(n, bweuler01, bweuler02)

m = 0
bweuler11, bweuler12 = 0, 0
while bweuler11 == bweuler12:
    m += 1
    bweuler11 = cv2.connectedComponents(IMAGE.astype(np.uint8))[0]
    element_structurant = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n+m,n+m))
    IMAGE_OPEN = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, element_structurant)
    bweuler12 = cv2.connectedComponents(IMAGE_OPEN.astype(np.uint8))[0]
    print(n+m, bweuler11, bweuler12)
    


