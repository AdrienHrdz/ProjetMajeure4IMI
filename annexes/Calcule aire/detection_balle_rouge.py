#from sys import last_value
from distutils.ccompiler import CCompiler
import cv2, platform,time
import matplotlib.pyplot as plt
import numpy as np

# 0 = Use  local webcam.
# 2 = Use extern webcam
cam = 2


S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
S2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
cap = cv2.VideoCapture(cam)
if not cap:
    print("!!! Failed VideoCapture: invalid parameter!")

while(True):
    # Capture frame-by-frame
    ret, current_frame = cap.read()
    
    if type(current_frame) == type(None):
        print("!!! Couldn't read frame!")
        break

    #division image
    frameB= current_frame[:,:,0]
    frameG= current_frame[:,:,1]
    frameR= current_frame[:,:,2] 


    frameB = cv2.threshold(frameB, 50, 255, cv2.THRESH_BINARY)
    frameG = cv2.threshold(frameG, 50, 255, cv2.THRESH_BINARY)
    frameR = cv2.threshold(frameR, 100,255, cv2.THRESH_BINARY)

    #reconstruction image
    frame2 =frameR[1] - frameG[1] - frameB[1]
    

    # Display the resulting frame

    
    Imod=cv2.morphologyEx(frame2,cv2.MORPH_OPEN,S)
    #cv2.imshow('final',Imod)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    #Allow to send only one frame per second
    #time.sleep(1)
    #if cv2.morphologyEx(Imod,cv2.MORPH_DILATE,S2):
    #    print("c'est bon")

    [N,labels]=cv2.connectedComponents(Imod)
    #print("le nombre d'élément est : ", N)
    Imod2=cv2.morphologyEx(Imod,cv2.MORPH_CLOSE,S2)
    [N2,labels2]=cv2.connectedComponents(Imod2)
    #res après fermeture
    cv2.imshow('final2',Imod2)
    print("le nombre d'élément est après fermeture : ", N2)
    if N2==2 :
        #print("c'est bon")
        cnt, hierarchy=cv2.findContours(Imod2,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        #cv2.drawContours(Imod2,cnt,-1,(0,0,255),5)
        #cv2.imshow('final3',Imod2)
        #print(cnt)
        #cnt = cnt.astype(np.uint32)
        cnt2=cnt[0]
        area= cv2.contourArea(cnt2)
        print("l'aire est de  : ",area)
        if (area<=306085.0) :
            if area>=250000.0:
                print("objet detecte") 
                #cv2.imwrite('image_fin.png',Imod2)
                #cap.release()
                #cv2.destroyAllWindows()
# release the capture



cap.release()
cv2.destroyAllWindows()
