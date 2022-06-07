from pickletools import uint8
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch import int64, int8

# plt.figure(1)
# #importation de l'image a étudié
# I=cv2.imread('data/color2.jpg',0)#.astype(float)
# plt.subplot(221)
# #prétraitement de l'image
# plt.imshow(cv2.cvtColor(I,cv2.COLOR_BGR2RGB))
# ret,thresh1=cv2.threshold(I,125,255,cv2.THRESH_BINARY)
# plt.subplot(222)
# plt.imshow(cv2.cvtColor(thresh1,cv2.COLOR_BGR2RGB))

# S=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

# I_open=cv2.dilate(cv2.erode(thresh1,S),S)
# plt.subplot(223)
# plt.imshow(cv2.cvtColor(I_open,cv2.COLOR_BGR2RGB))
# #afin d'avoir le bon type
# TYPE=type(I)

# IMAGE=I_open.astype(TYPE)
# IMAGE2=I_open
# #partie snake

# Lx, Ly = np.shape(IMAGE2)

# # %%
# ###Creation du snake###
# centre=[int(Ly/2),int(Lx/2)]
# rayon=min(int((Lx-5)/2), int((Ly-5)/2))/3

# K = 1000
# snakeX = []
# snakeY = []
# pas = (2*np.pi)/K
# for i in range(K):
#     theta = i*pas
#     snakeX = np.append(snakeX, int(centre[0] + rayon * np.cos(theta)))
#     snakeY = np.append(snakeY, int(centre[1] + rayon * np.sin(theta)))
# # print(snakeX.shape)
# c = np.zeros((K,1,2))
# # print(c.shape)
# c[:,:,0] = snakeX.reshape((K,1))
# c[:,:,1] = snakeY.reshape((K,1))
# #c = np.concatenate((snakeX.reshape((K,1)),snakeY.reshape((K,1))),axis=2)
# # print(c[:,0,0])
# contour_list = []
# contour_list.append(c.astype(int))
# snake = cv2.drawContours(image=cv2.cvtColor(IMAGE2, cv2.COLOR_GRAY2BGR),contours=contour_list, contourIdx=-1, color=(255, 0, 0), thickness=1,lineType=cv2.LINE_AA)
# plt.imshow(snake)
# plt.show()



# # %%
# ### Parametres ###
# alpha = 3
# beta = 0.1
# gamma = 1.2

# # %% [markdown]
# # #### On défini les matrices pour les opérateurs
# # La matrice `D1` correspond à une approximation de la dérivée par différence finies. La matrice `D2` correspond à une dérivée seconde et `D4` à une dérivée quatrième.

# # %%
# ###Creation de D2, D4, D et A###
# Id = np.identity(K)
# D1 = np.roll(Id, 1, axis=-1) + Id*(0) - np.roll(Id,-1, axis=1)
# D2 = np.roll(Id, -1, axis=1) + Id*(-2) + np.roll(Id,1, axis=1)
# D4 = (np.roll(Id, -1, axis=1) + np.roll(Id,1, axis=1))*-4 + (np.roll(Id, -2, axis=1) + np.roll(Id,2, axis=1)) + Id*(6)
# D = alpha*D2 - beta*D4
# A = np.linalg.inv(Id - D)
# #logging.info('Operators generated')

# # %%
# # Le Gradient
# [Gx,Gy] = np.gradient(IMAGE2.astype(float))
# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(Gx)
# plt.subplot(1,3,2)
# plt.imshow(Gy)
# Gx_norm = Gx/np.max(Gx)
# Gy_norm = Gy/np.max(Gy)
# NormeGrad = np.square(Gx_norm)+np.square(Gy_norm)

# plt.subplot(1,3,3)
# plt.imshow(NormeGrad,'gray')
# plt.show()
# #NormeGrad = NormeGrad*20
# # Gradient de la norme 
# [GGx,GGy] = np.gradient(NormeGrad.astype(float))

# #logging.info('Gradient computed')

# # %%
# # Algo ITERATIF
# limite = 3000
# iteration = 0
# nbfigure = 1

# Energie = list()
# energie_ela = list()
# energie_courb = list()
# enregie_ext = list()

# MEMORY = []
# Xn = snakeX
# Yn = snakeY
# MEMORY.append([Xn,Yn])
# #logging.info('Algorithm initialized')

# # %%
# flag = True
# while flag or (iteration < limite):
#     # itération du SNAKE
#     Xn1 = np.dot(A, Xn + gamma*GGx[Yn.astype(int),Xn.astype(int)] )
#     Yn1 = np.dot(A, Yn + gamma*GGy[Yn.astype(int),Xn.astype(int)] )     
#     Xn = Xn1
#     Yn = Yn1   
#     MEMORY.append([Xn,Yn])
#     # Calcul de l'energie
#     ELA = 0
#     COURB  = 0
#     EXT = 0
#     Xnprime = np.dot(D1, Xn)
#     Ynprime = np.dot(D1, Yn)
#     Xnseconde = np.dot(D2, Xn)
#     Ynseconde = np.dot(D2, Yn)
#     for k in range(K):
#         ELA += alpha*0.5*np.sqrt(np.square(Xnprime[k]) + np.square(Ynprime[k]))
#         COURB += beta*0.5*np.sqrt(np.square(Xnseconde[k]) + np.square(Ynseconde[k]))
#         EXT -= np.square(NormeGrad[int(Yn[k]),int(Xn[k])])
#     Energie.append(ELA+COURB+EXT)
#     enregie_ext.append(EXT)
#     energie_courb.append(COURB)
#     energie_ela.append(ELA)

#     # Flag de sortie
#     # 
#     # TODO : 
#     #   Calculer energie sur fenetre glissante pour flag de sortie afin de lisser les variations
#     # if iteration > 200:
#     #     nbSplit = iteration // 50
#     #     EnerSplit = np.split(Energie, nbSplit)
#     #     e1 = EnerSplit[-1]
#     #     e2 = EnerSplit[-2]
#     if (abs(Energie[iteration]-Energie[iteration-1])/Energie[iteration]<10):
#         flag = False

    
#     # Affichage
#     if iteration % 10 == 0:
#         c = np.zeros((K,1,2))
#         c[:,:,0] = Xn1.reshape((K,1))
#         c[:,:,1] = Yn1.reshape((K,1))
#         contour_list = []
#         contour_list.append(c.astype(int))
#         snake = cv2.drawContours(image=cv2.cvtColor(IMAGE2, cv2.COLOR_GRAY2BGR),contours=contour_list, contourIdx=-1, color=(255, 0, 0), thickness=1,lineType=cv2.LINE_AA)
#         # Sauvegarde des images pour faire l'animation
#         #filename = f"img_{iteration:05d}.png"
        
#         #cv2.imwrite(filename, snake)

#     # Fin de la boucle
#     iteration += 1

# # %%
# c = np.zeros((K,1,2))
# print(c.shape)
# c[:,:,0] = Xn1.reshape((K,1))
# c[:,:,1] = Yn1.reshape((K,1))
# contour_list = []
# contour_list.append(c.astype(int))
# snake = cv2.drawContours(image=cv2.cvtColor(IMAGE2, cv2.COLOR_GRAY2BGR),contours=contour_list, contourIdx=-1, color=(255, 0, 0), thickness=cv2.FILLED,lineType=cv2.LINE_AA)
# plt.imshow(snake)
# cv2.imwrite("itération finale.png",snake)
# plt.title('Itération finale')
# plt.show()

I_original=cv2.imread('data/gdb_benin.jpg')

I_flout = cv2.blur(I_original,(9,9))

plt.figure()
plt.imshow(cv2.cvtColor(I_flout,cv2.COLOR_BGR2RGB))
plt.show()

b,g,r = cv2.split(I_flout)

snake_import = cv2.imread('itération finale.png')

snakegray = cv2.cvtColor(snake_import, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(snakegray)
plt.title('Snake Gray')
plt.show()

ret,I_seuilinv = cv2.threshold(snakegray,254,255,cv2.THRESH_BINARY_INV)

plt.figure()
plt.imshow(I_seuilinv)
plt.show()

I_masque_b = cv2.multiply(I_seuilinv, b)

I_masque_g = cv2.multiply(I_seuilinv, g)

I_masque_r = cv2.multiply(I_seuilinv, r)


I_masque = np.zeros((300,360,3))
I_masque = I_masque.astype(np.uint8)

I_masque[:,:,0] = I_masque_b
I_masque[:,:,1] = I_masque_g
I_masque[:,:,2] = I_masque_r

#I_masque = cv2.cvtColor(I_masque, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(I_masque)
plt.show()

I_masque[I_masque == 255] = 1

I_fin = cv2.multiply(I_masque, I_flout)

plt.figure()
plt.imshow(cv2.cvtColor(I_fin, cv2.COLOR_BGR2RGB))
plt.show()


lenx, leny, dim = np.shape(I_fin)

Liste=[]
for i in range(lenx):
    for j in range(leny):
            if(I_fin[i,j,0] != 0 or I_fin[i,j,1] != 0 or I_fin[i,j,2] != 0 ):
                Liste.append([i,j])

#Faire la moyenne des pixels de couleur (grain boté) et faire la différence entre les 2 kmeans et la moyenne




#Sinon autre façon de faire => Parcourir tous les pixels de l'image récuperer leur valeurs de couleur (r,g,b) 
# puis en fonction faire des pourcentages





#I_flout = cv2.blur(I_original,(13,13))

#lenx, leny, dim = np.shape(I_flout)

RGB_img = cv2.cvtColor(I_flout, cv2.COLOR_BGR2RGB)

HSV_img = cv2.cvtColor(I_flout, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(HSV_img)

lenx, leny = np.shape(h)

#Bon réglages pour color.jpg
lower = np.array([14/2,100,0],dtype=np.uint8)
upper = np.array([60/2,255,255],dtype=np.uint8)
seg_h = cv2.inRange(HSV_img,lower,upper)

#Idée post-traitement compare l'aire du grain de beauté à l'aire de la tâche de couleur si celle ci est trop importante alors malade

#Réglages pour grainboTbien
# lower = np.array([0/2,83,0],dtype=np.uint8)
# upper = np.array([14/2,255,255],dtype=np.uint8)
# seg_h = cv2.inRange(HSV_img,lower,upper)
# S=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
# seg_h_open = cv2.morphologyEx(seg_h,cv2.MORPH_OPEN,S)

I_cropped = cv2.imread('data/Capture_fig4_test2.PNG',0)





I_cropped2 = RGB_img[108:185,140:220]

lenx2, leny2, dim= np.shape(I_cropped2)

#I_resize= cv2.resize(I_cropped,(360,300),interpolation=cv2.INTER_AREA)

plt.figure() # ouvre une nouvelle figure
plt.subplot(221)
plt.imshow(RGB_img) 
plt.title('Image original RGB')
plt.subplot(222)
plt.imshow(I_cropped,'gray')
plt.title('Image H ')
plt.subplot(223)
plt.imshow(s,'gray') 
plt.title('Image S ')
plt.subplot(224)
plt.imshow(I_cropped2,'gray') 
plt.title('Image Seg H ')
plt.show()

#Méthode des K-means
#Potentiellement fonctionnel pour color et color3
#Si la différence de couleur entre le dernier k (grain de beauté) et l'avant dernier (grain de beauté ou tâche) est trop importante (à quantifier) alors => pas bien

#Tableau bi-dimensionnel

pixel_vals1 = h.reshape((-1,1)) 
  
pixel_vals2 = I_fin.reshape((-1,3)) 

pixel_vals1 = np.float32(pixel_vals1)

pixel_vals2 = np.float32(pixel_vals2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.8) 
  
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals2, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 
  
centers = np.uint8(centers) 
segmented_data = centers[labels.flatten()] 

#labels_img = labels.reshape((lenx,leny)) 
labels_img = labels.reshape((lenx,leny)) 

segmented_image = segmented_data.reshape((I_fin.shape)) 

Coords1 = []
Coords2 = []
bool1=1
bool2=1
for i in range(lenx):
    for j in range(leny):
        if(labels_img[i,j] == 0 and bool1 == 1):
            Coords1.append(i)
            Coords1.append(j)
            bool1=0
        if(labels_img[i,j]== 1 and bool2 == 1):
            Coords2.append(i)
            Coords2.append(j)
            bool2=0

#[r1,g1,b1] = segmented_image[Coords1[0],Coords1[1]]
#[r2,g2,b2] = segmented_image[Coords2[0],Coords2[1]]

teinte1=segmented_image[Coords1[0],Coords1[1]]
teinte2=segmented_image[Coords2[0],Coords2[1]]


plt.figure() # ouvre une nouvelle figure
plt.imshow(segmented_image)
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





