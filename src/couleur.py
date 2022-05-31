import cv2
import matplotlib.pyplot as plt
import numpy as np


# I_original=cv2.imread('data/color.jpg', 0).astype(float)
# ret,I_seuil=cv2.threshold(I_original,50,255,cv2.THRESH_BINARY) # pour le seuillage d'image
# I_calc = cv2.multiply(I_seuil,I_original)
# #I_calc =  I_seuilinv - I_original 


#Sinon autre façon de faire => Parcourir tous les pixels de l'image récuperer leur valeurs de couleur (r,g,b) 
# puis en fonction faire des pourcentages


I_original=cv2.imread('data/color3.jpg')

I_flout = cv2.blur(I_original,(15,15))

I_original_B = I_original[:,:,0]
I_original_G = I_original[:,:,1]
I_original_R = I_original[:,:,2]

RGB_img = cv2.cvtColor(I_flout, cv2.COLOR_BGR2RGB)

HSV_img = cv2.cvtColor(I_flout, cv2.COLOR_BGR2HSV)

## mask of red color
mask1 = cv2.inRange(I_original, (20, 20, 110), (140, 140,210)) #BRG


plt.figure() # ouvre une nouvelle figure
plt.subplot(221)
plt.imshow(RGB_img) 
plt.title('Image original')
plt.subplot(222)
plt.imshow(mask1,'gray')
plt.title('Image Mask ')
plt.subplot(223)
plt.imshow(HSV_img[:,:,0],'gray') 
plt.title('Image G ')
plt.subplot(224)
plt.imshow(HSV_img[:,:,1],'gray') 
plt.title('Image R ')
plt.show()




