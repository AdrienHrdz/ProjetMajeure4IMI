import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy as sp
from divers import LECTURE_IMAGE
import pypher

# filename = '../data/gdb_benin.jpg'
# z = LECTURE_IMAGE(filename)
z = cv2.imread('./data/gdb_benin.jpg',0).astype(float)
# z = cv2.cvtColor(z, cv2.COLOR_RGB2GRAY)
# z = z[2:-2, 2:-2]
plt.imshow(z, cmap='gray')
plt.title('Z')
plt.show()
(H,W) = np.shape(z)
lamb = 1.5
gamma = 0.005

## LAPLACIAN
def L(x):
    dim = (np.min(np.shape(x)) > 1) + 1
    if dim == 1:
        ker = [1,-2,1]
        # Faire zpadding 1D
        OTF = np.fft.fft(ker, np.size(x))
        lap = np.real(np.fft.ifft( OTF * np.fft.fft(x) ))
    elif dim == 2:
        ker = [[0,1,0],[1,-4,1],[0,1,0]] # V4
        # ker = [[1,1,1],[1,-8,1],[1,1,1]] # V8
        ker_padding = np.zeros(np.shape(x))
        (H,W) = np.shape(ker_padding)
        ker_padding[H//2 - 1:H//2 +2 , W//2 - 1:W//2 +2] = ker
        
        OTF = np.fft.fft2(ker, np.shape(x))
        lap = np.real(np.fft.ifft2( OTF * np.fft.fft2(x) ))
    return lap

## LAPLACIAN Transpose
def Lt(x):
    dim = (np.min(np.shape(x)) > 1) + 1
    if dim == 1:
        ker = [1,-2,1]
        # Faire zpadding 1D
        OTF = np.fft.fft(ker, np.size(x))
        lap = np.real(np.fft.ifft(np.conj(OTF * np.fft.fft(x))))
    elif dim == 2:
        ker = [[0,1,0],[1,-4,1],[0,1,0]] # V4
        # ker = [[1,1,1],[1,-8,1],[1,1,1]] # V8
        ker_padding = np.zeros(np.shape(x))
        (H,W) = np.shape(ker_padding)
        ker_padding[H//2 - 1:H//2 +2 , W//2 - 1:W//2 +2] = ker
        OTF = np.fft.fft2(ker, np.shape(x))
        lap = np.real(np.fft.ifft2(OTF * np.fft.fft2(x)))
    return lap

## Prox norme L0
def prox_tau_L0(x, tau):
    return (x>tau)*x + (x<-tau)*x

## Prox norme L1
def prox_tau_L1(x, tau):
    return (x>tau)*(x-tau) + (x<-tau)*(x+tau) 

G = L
Gt = Lt

def gradF(u):
    grad = -G(-Gt(u)+z)
    return grad

Niter = 1e3
iter = 0
ENERGIE = list()
uk = G(z)
while iter < Niter:
    uk1 = uk - gamma*gradF(uk) - gamma*prox_tau_L0( uk/gamma - gradF(uk), lamb/gamma)
    # faire calcul énergie pour voir si on a convergé
    cout = 0.5*np.linalg.norm(-Gt(uk1)-z, "fro")**2 + lamb*np.linalg.norm(uk1, 1)
    ENERGIE.append(cout)
    uk = uk1
    # print(iter)
    iter += 1

uhat = uk
xhat = -Gt(uhat) + z
plt.figure()
plt.imshow(xhat, cmap='gray')
plt.figure()
plt.loglog(ENERGIE)
plt.show()



