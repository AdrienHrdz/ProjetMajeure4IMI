import cv2
import matplotlib.pyplot as plt
import numpy as np


# I_original=cv2.imread('data/color.jpg', 0).astype(float)
# ret,I_seuil=cv2.threshold(I_original,50,255,cv2.THRESH_BINARY) # pour le seuillage d'image
# I_calc = cv2.multiply(I_seuil,I_original)
# #I_calc =  I_seuilinv - I_original 


#Sinon autre façon de faire => Parcourir tous les pixels de l'image récuperer leur valeurs de couleur (r,g,b) 
# puis en fonction faire des pourcentages

#Fonction trouvée sur un forum
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

#grainboTbien
I_original=cv2.imread('data/gdb_benin.jpg')

#I_original=crop_and_resize(I_original,360,300)



I_flout = cv2.blur(I_original,(13,13))

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

I_resize2 = image_resize(I_cropped,width = 360)

I_fin = cv2.resize(I_resize2, (360,300), interpolation = cv2.INTER_AREA)

#I_resize= cv2.resize(I_cropped,(360,300),interpolation=cv2.INTER_AREA)

plt.figure() # ouvre une nouvelle figure
plt.subplot(221)
plt.imshow(RGB_img) 
plt.title('Image original RGB')
plt.subplot(222)
plt.imshow(I_fin,'gray')
plt.title('Image H ')
plt.subplot(223)
plt.imshow(s,'gray') 
plt.title('Image S ')
plt.subplot(224)
plt.imshow(seg_h,'gray') 
plt.title('Image Seg H ')
plt.show()

#Méthode des K-means
#Potentiellement fonctionnel pour color et color3
#Si la différence de couleur entre le dernier k (grain de beauté) et l'avant dernier (grain de beauté ou tâche) est trop importante (à quantifier) alors => pas bien

#Tableau bi-dimensionnel

pixel_vals1 = h.reshape((-1,1)) 
  
pixel_vals2 = RGB_img.reshape((-1,3)) 

pixel_vals1 = np.float32(pixel_vals1)

pixel_vals2 = np.float32(pixel_vals2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.8) 
  
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals1, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 
  
centers = np.uint8(centers) 
segmented_data = centers[labels.flatten()] 

labels_img = labels.reshape((lenx,leny)) 

segmented_image = segmented_data.reshape((h.shape)) 

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





