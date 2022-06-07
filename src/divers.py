import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt


def LECTURE_IMAGE(filename):
    # doc : 
    # https://linuxtut.com/fr/c6e468da7007734c897f/

    p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    img = cv2.imread(f'{filename}',0)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, p_dict) #détection

    m = np.empty((4,2))
    corners2 = [np.empty((1,4,2))]*4
    for i,c in zip(ids.ravel(), corners):
        corners2[i] = c.copy()

    m[0] = corners2[0][0][2]
    m[1] = corners2[1][0][3]
    m[2] = corners2[2][0][0]
    m[3] = corners2[3][0][1]

    width, height = (501,501) #Taille de l'image après transformation
    marker_coordinates = np.float32(m)
    true_coordinates   = np.float32([[0,0],[width,0],[width,height],[0,height]])
    trans_mat = cv2.getPerspectiveTransform(marker_coordinates,true_coordinates)
    img_trans = cv2.warpPerspective(img,trans_mat,(width, height))
    img_trans = cv2.cvtColor(img_trans, cv2.COLOR_BGR2RGB)

    return img_trans

def main():
    filename = './aruco./IMG_20220531_170740.jpg'
    IMAGE = LECTURE_IMAGE(filename)
    plt.imshow(IMAGE, cmap='gray')
    plt.show()
    pass

if __name__ == '__main__':
    main()
