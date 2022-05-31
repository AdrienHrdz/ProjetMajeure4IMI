from re import I
import cv2, platform,time
from cv2 import waitKey
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

plt.figure(1)

I=cv2.imread('data/gdb_benin.jpg',0)#.astype(float)
plt.subplot(221)
plt.imshow(cv2.cvtColor(I,cv2.COLOR_BGR2RGB))
ret,thresh1=cv2.threshold(I,125,255,cv2.THRESH_BINARY)
plt.subplot(222)
plt.imshow(cv2.cvtColor(thresh1,cv2.COLOR_BGR2RGB))

S=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

I_open=cv2.dilate(cv2.erode(thresh1,S),S)
plt.subplot(223)
plt.imshow(cv2.cvtColor(I_open,cv2.COLOR_BGR2RGB))
I=I_open

#partie snake

def moyenne(liste):
    return sum(liste)/len(liste)

li,col = len(I),len(I[0])
alpha = 1
beta = 0.5
gamma = 15
centre=[int(col/2),int(li/2)]
rayon=min(int((col-5)/2),int((li-5)/2))
delta = 1.
K = 1000
snakeX,snakeY = [],[]
pas = (2*np.pi)/K
for i in range(K):
    teta = i*pas
    snakeX = np.append(snakeX, int(centre[0] + rayon * np.cos(teta)))
    snakeY = np.append(snakeY, int(centre[1] + rayon * np.sin(teta)))


Id = np.identity(K)
D1 = np.roll(Id, 1, axis=-1) + Id*(0) - np.roll(Id,-1, axis=1)
D2 = np.roll(Id, -1, axis=1) + Id*(-2) + np.roll(Id,1, axis=1)
D4 = (np.roll(Id, -1, axis=1) + np.roll(Id,1, axis=1))*-4 + (np.roll(Id, -2, axis=1) + np.roll(Id,2, axis=1)) + Id*(6)
D = alpha*D2 - beta*D4
A = np.linalg.inv(Id - D)


Gy,Gx = np.gradient(I.astype(float))
NormGrad = Gx**2 + Gy**2
GGy, GGx = np.gradient(NormGrad)

NRJ,NRJELA,NRJCOURB,NRJEXT = [],[],[],[] 
MOY,MINMAX,DELTA = [],[],[]
GxSnake = np.zeros(snakeX.shape)
GySnake = np.zeros(snakeY.shape)
it=0  # nombre d'itération
flag = True
j=0

limite = 10000


while flag and it<limite:
    for i in range(K):
        Y=int(snakeY[i])
        X=int(snakeX[i])
        GxSnake[i] = GGx[Y][X]
        GySnake[i] = GGy[Y][X]
    snakeX = np.dot(A, snakeX+gamma*GxSnake)
    snakeY = np.dot(A, snakeY+gamma*GySnake)
    # Calcul de l'energie
    ELA,COURB,EXT = 0,0,0
    Xnprime = np.dot(D1, snakeX)
    Ynprime = np.dot(D1, snakeY)
    Xnseconde = np.dot(D2, snakeX)
    Ynseconde = np.dot(D2, snakeY)
    for k in range(K):
        ELA += alpha*0.5*np.sqrt(np.square(Xnprime[k]) + np.square(Ynprime[k]))
        COURB += beta*0.5*np.sqrt(np.square(Xnseconde[k]) + np.square(Ynseconde[k]))
        EXT += NormGrad[int(snakeY[k]),int(snakeX[k])]**2
    NRJ.append(ELA+COURB-EXT)
    NRJEXT.append(EXT)
    NRJCOURB.append(COURB)
    NRJELA.append(ELA)

    if it>300:
        delta1 = [NRJ[it-i] for i in range(250)]
        MOY.append(moyenne(delta1))
    it+=1



plt.figure()
plt.imshow(I,'gray')
plt.plot(snakeX, snakeY, 'r', linewidth=1)
plt.text(col/6,15,"Alpha = "+str(alpha)+" ; Beta = "+str(beta)+" ; Gamma = "+str(gamma))
plt.text(col/3,25,"Pour "+str(it)+" itérations")



plt.show()
