import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal as signal
from divers import LECTURE_IMAGE


## Id
def Id(x):
    return x

## Id Transpose
def Idt(x):
    return x

## GRADIENT
def D(x):
    dim = (np.min(np.shape(x)) > 1) + 1
    if dim == 1:
        pass
    elif dim == 2:
        ker = np.array([[0,1,0], [1,-4,1], [0,1,0]], dtype=np.float64)
        lap = signal.convolve2d(x, ker, mode='same', boundary='symm')
    return lap

## GRADIENT Transpose
def Dt(x):
    dim = (np.min(np.shape(x)) > 1) + 1
    if dim == 1:
        pass
    elif dim == 2:
        ker = np.array([[0,1,0], [1,-4,1], [0,1,0]], dtype=np.float64)
        lap = signal.convolve2d(x, ker, mode='same', boundary='symm')
    return lap

## LAPLACIAN
def L(x):
    dim = (np.min(np.shape(x)) > 1) + 1
    if dim == 1:
        pass
    elif dim == 2:
        ker = np.array([[0,1,0], [1,-4,1], [0,1,0]], dtype=np.float64)
        lap = signal.convolve2d(x, ker, mode='same', boundary='symm')
    return lap

## LAPLACIAN Transpose
def Lt(x):
    dim = (np.min(np.shape(x)) > 1) + 1
    if dim == 1:
        pass
    elif dim == 2:
        ker = np.array([[0,1,0], [1,-4,1], [0,1,0]], dtype=np.float64)
        lap = signal.convolve2d(x, ker, mode='same', boundary='symm')
    return lap

## Prox norme L0
def prox_tau_L0(x, tau):
    return (x>tau)*x + (x<-tau)*x

## Prox norme L1
def prox_tau_L1(x, tau):
    return (x>tau)*(x-tau) + (x<-tau)*(x+tau) 


def total_variation(z, lamb, gamma, operator, Niter):
    # Choix de l'opérateur
    if operator == 'id':
        G = Id
        Gt = Idt
    elif operator == 'gradient':
        G = D
        Gt = Dt
    elif operator == 'laplacian':
        G = L
        Gt = Lt

    def gradF(u):
        grad = -G(-Gt(u)+z)
        return grad

    # initialisation
    iter = 0
    ENERGIE = list()
    uk = G(z)

    # boucle
    while iter < Niter:
        uk1 = uk - gamma*gradF(uk) - gamma*prox_tau_L0( uk/gamma - gradF(uk), lamb/gamma)
        # faire calcul énergie pour voir si on a convergé
        cout = 0.5*np.linalg.norm(-Gt(uk1)-z, "fro")**2 + lamb*np.linalg.norm(uk1, 1)
        ENERGIE.append(cout)
        uk = uk1
        iter += 1

    # post traitement
    uhat = uk
    xhat = -Gt(uhat) + z
    return xhat, ENERGIE

def main():
    z = cv2.imread('./data/gdb_benin.jpg',0).astype(float)
    plt.imshow(z, cmap='gray')
    plt.title('Z')
    
    lamb = 100
    gamma = 0.005
    Niter = 1e3
    operator = 'laplacian'
    xhat, ENERGIE = total_variation(z, lamb, gamma, operator, Niter)

    plt.figure()
    plt.imshow(xhat, cmap='gray')
    plt.figure()
    plt.loglog(ENERGIE)
    
    (Gx, Gy) = np.gradient(xhat)
    NormCarre = np.square(Gx) + np.square(Gy)
    plt.figure()
    plt.imshow(NormCarre, cmap='gray')
    plt.show()
    

if __name__ == '__main__':
    main()

