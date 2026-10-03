# Wiener Algorithm
# Var of Noise with the Original to get the Correct one
#original image minus noise sigma
esp=1e-4

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import SmoothFunctions as SF
import FourierFilter as FF


pic = cv.imread('puppy.jpg', cv.IMREAD_GRAYSCALE)

def Hest(hkern,shape):
    M,N=shape
    kh,kw=hkern.shape
    Hpad=np.zeros((M,N),dtype=np.float64)
    r_center, c_center = M // 2, N // 2
    
    r_start = r_center - kh // 2
    c_start = c_center - kw // 2
    Hpad[r_start : r_start + kh, c_start : c_start + kw] = hkern / np.sum(hkern)
    #Hpad[:kh,:kw]=hkern/np.sum(hkern)
    
    
    H=np.fft.fft2(np.fft.ifftshift(Hpad))
    return H

def PatchSigmaEst(image,xpoint,ypoint,width,height):
    patch=image[ypoint:ypoint+height,xpoint:xpoint+width]
    sigma_n = np.std(patch)
    sigma_n_sq = np.var(patch)
    return sigma_n,sigma_n_sq

def SigmaEstGen(image):
    img = image.astype(np.float64)
    H,W=img.shape
    M = (
        img[:-2, :-2] - 2*img[:-2, 1:-1] + img[:-2, 2:]
      - 2*img[1:-1, :-2] + 4*img[1:-1, 1:-1] - 2*img[1:-1, 2:]
      + img[2:, :-2] - 2*img[2:, 1:-1] + img[2:, 2:]
    )
    sum_abs = np.sum(np.abs(M))
    scale = np.sqrt(0.5 * np.pi) / (6.0 * (W - 2) * (H - 2))
    sigma_n = scale * sum_abs
    sigma_n_sq = sigma_n ** 2
    return sigma_n, sigma_n_sq

def Wiener(image,hkern,sigma2):
    rw,cl=image.shape[:2]
    form=(rw,cl)
    H=Hest(hkern,form)

    G=np.fft.fft2(image)
    Hconj=np.conj(H)
    H2=np.abs(H)**2

    Sg=(np.abs(G) ** 2) / (rw * cl)
    Sf=np.maximum(Sg-sigma2,0.0)/(H2+esp)

    NSR=np.where(Sf>0,sigma2/(Sf+esp),sigma2)
    W=Hconj/(H2+NSR)

    Fh=W*G
    restore=np.real(np.fft.ifft2(Fh))

    return np.clip(restore, 0, 255)

sigma=1
noise=0.02

kernel_size = 3
k1d=SF.GK(3,sigma)

hkern = k1d @ k1d.T

Damage_Image=SF.GaussSmooth((FF.SPnoise(pic,noise)),3,sigma)
Cleaned=SF.MediumFilter(Damage_Image.astype(np.uint8), 3)
SigmaEst,SigmaEst2 = PatchSigmaEst(Cleaned,0,0,10,10)

print(SigmaEst)
print(SigmaEst2)
Restore=Wiener(Cleaned,hkern,SigmaEst2)
plt.subplot(1, 2, 1)
plt.title("Damaged")
plt.imshow(Damage_Image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Restored")
plt.imshow(Restore, cmap="gray")
plt.axis("off")
plt.show()

