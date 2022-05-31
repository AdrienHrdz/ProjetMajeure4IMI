from re import I
import cv2, platform,time
from cv2 import waitKey
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg



I=cv2.imread('data/gdb_benin.jpg',0)#.astype(float)

plt.imshow(cv2.cvtColor(I,cv2.COLOR_BGR2RGB))

#cv2.imshow('image',I)
#waitKey(0)






plt.show()
