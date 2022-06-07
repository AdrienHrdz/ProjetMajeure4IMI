import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy as sp
from divers import LECTURE_IMAGE
from pypher import psf2otf

filename = './aruco./IMG_20220531_170740.jpg'
z = LECTURE_IMAGE(filename)
z = cv2.cvtColor(z, cv2.COLOR_RGB2GRAY)
plt.imshow(z, cmap='gray')
plt.show()
z = z[2:-2, 2:-2]
plt.imshow(z, cmap='gray')
plt.show()
(H,W) = np.shape(z)
lamb = 50
gamma = 0.05

## LAPLACIAN
def L(x):
    dim = (np.min(np.shape(x)) > 1) + 1
    if dim == 1:
        ker = [1,-2,1]
        lap = np.real(np.ifft( psf2otf(ker, np.shape(x)) * np.fft(x) ))
    elif dim == 2:
        ker = [[0,1,0],[1,-4,1],[0,1,0]] # V4
        # ker = [[1,1,1],[1,-8,1],[1,1,1]] # V8
        lap = np.real(np.ifft2( psf2otf(ker, np.shape(x)) * np.fft2(x) ))
    return lap

## LAPLACIAN Transpose
def Lt(x):
    dim = (np.min(np.shape(x)) > 1) + 1
    if dim == 1:
        ker = [1,-2,1]
        lap = np.real(np.ifft(np.conj(psf2otf(ker, np.shape(x))) * np.fft(x)))
    elif dim == 2:
        ker = [[0,1,0],[1,-4,1],[0,1,0]] # V4
        # ker = [[1,1,1],[1,-8,1],[1,1,1]] # V8
        lap = np.real(np.ifft2(np.conj(psf2otf(ker, np.shape(x))) * np.fft2(x)))
    return lap

## Prox norme L0
def prox_tau_L0(x, tau):
    return (x>tau)*x + (x<-tau)*x

# Prox norme L1
def prox_tau_L1(x, tau):
    return (x>tau)*(x-tau) + (x<-tau)*(x+tau) 

G = L
Gt = Lt

def gradF(u):
    grad = -G(-Gt(u)+z)
    return grad

Niter = 1e4
iter = 0
uk = G(z)
while iter < Niter:
    uk1 = uk - gamma*gradF(uk) - gamma*prox_tau_L0( uk/gamma - gradF(uk), lamb/gamma)
    uk = uk1
    iter += 1

uhat = uk
xhat = -Gt(uhat) + z



