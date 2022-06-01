import cv2
import numpy as np
import matplotlib.pyplot as plt

from divers import LECTURE_IMAGE

filename = './testBords/IMG_20220601_114342.jpg'

IMAGE = LECTURE_IMAGE(filename)
IMAGE = cv2.cvtColor(IMAGE, cv2.COLOR_RGB2GRAY)

plt.imshow(IMAGE, cmap='gray')
plt.show()

[Gx,Gy] = np.gradient(IMAGE.astype(float)) # normaliser 
NormGrad = np.sqrt(np.square(Gx) + np.square(Gy))

plt.imshow(NormGrad, cmap='gray')
plt.colorbar()
plt.show()