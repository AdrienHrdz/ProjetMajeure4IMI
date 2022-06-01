import cv2
import matplotlib.pyplot as plt
import numpy as np


# I_original=cv2.imread('data/color.jpg', 0).astype(float)
# ret,I_seuil=cv2.threshold(I_original,50,255,cv2.THRESH_BINARY) # pour le seuillage d'image
# I_calc = cv2.multiply(I_seuil,I_original)
# #I_calc =  I_seuilinv - I_original 


#Sinon autre façon de faire => Parcourir tous les pixels de l'image récuperer leur valeurs de couleur (r,g,b) 
# puis en fonction faire des pourcentages

#grainboTbien
I_original=cv2.imread('data/grainboTbien.jpg')

I_flout = cv2.blur(I_original,(13,13))

I_original_B = I_original[:,:,0]
I_original_G = I_original[:,:,1]
I_original_R = I_original[:,:,2]

RGB_img = cv2.cvtColor(I_flout, cv2.COLOR_BGR2RGB)

HSV_img = cv2.cvtColor(I_flout, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(HSV_img)

#Bon réglages pour color.jpg
# lower = np.array([14/2,100,0],dtype=np.uint8)
# upper = np.array([60/2,255,255],dtype=np.uint8)
# seg_h = cv2.inRange(HSV_img,lower,upper)

#Idée post-traitement compare l'aire du grain de beauté à l'aire de la tâche de couleur si celle ci est trop importante alors malade

#Réglages pour grainboTbien
lower = np.array([0/2,83,0],dtype=np.uint8)
upper = np.array([14/2,255,255],dtype=np.uint8)
seg_h = cv2.inRange(HSV_img,lower,upper)
S=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
seg_h_open = cv2.morphologyEx(seg_h,cv2.MORPH_OPEN,S)



plt.figure() # ouvre une nouvelle figure
plt.subplot(221)
plt.imshow(RGB_img) 
plt.title('Image original RGB')
plt.subplot(222)
plt.imshow(h,'gray')
plt.title('Image H ')
plt.subplot(223)
plt.imshow(s,'gray') 
plt.title('Image S ')
plt.subplot(224)
plt.imshow(seg_h_open,'gray') 
plt.title('Image Seg H ')
plt.show()




## mask of red color pour RGB
#mask1 = cv2.inRange(I_flout, (20, 20, 110), (140, 140,210)) #BRG

#maskmarron = cv2.inRange(HSV_img, (14, 95, 98), (52, 95,98))


#maskred = cv2.inRange(HSV_img, (212, 95, 98), (264, 95,98))

#mask = cv2.bitwise_or(maskgreen, maskred)

#target = cv2.bitwise_and(I_flout,I_flout, mask=mask)

#Masque pour HSV
#mask2 = cv2.inRange(HSV_img, (136, 87, 111), (180, 255, 255)) 
#ret,I_seuil=cv2.threshold(HSV_img[:,:,0],5,255,cv2.THRESH_BINARY)

# red_lower = np.array([20, 20, 10], np.uint8) 
# red_upper = np.array([140, 140, 210], np.uint8) 
# red_mask = cv2.inRange(HSV_img, red_lower, red_upper)

# kernal = np.ones((5, 5))

#red_mask = cv2.dilate(red_mask, kernal) 
#res_red = cv2.bitwise_and(I_flout, I_flout,mask = red_mask)





