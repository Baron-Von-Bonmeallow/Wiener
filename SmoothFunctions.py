import matplotlib.pyplot as plt
import numpy as np
import cv2
path=r"C:\Users\USUARIO\Documents\GitHub\Homework-Codes"
# C:\Users\USUARIO\Documents\GitHub\Homework-Codes
# c:\Users\memip\Documents\GitHub\Code

K1=np.array([
    [0,1,0],
    [1,-4,1],
    [0,1,0]
])
K2=np.array([
    [1,1,1],
    [1,-8,1],
    [1,1,1]
])
K3=np.array([
    [0,-1,0],
    [-1,4,-1],
    [0,-1,0]
])
K4=np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])

Klist=[K1,K2,K3,K4]

def MS(array):
    if len(array) > 1:
        mid = len(array) // 2
        lf = array[:mid].copy()
        rg = array[mid:].copy()
        MS(lf)
        MS(rg)
        i = j = k = 0
        while i < len(lf) and j < len(rg):
            if lf[i] <= rg[j]:
                array[k] = lf[i]
                i += 1
            else:
                array[k] = rg[j]
                j += 1
            k += 1
        while i < len(lf):
            array[k] = lf[i]
            i += 1
            k += 1
        while j < len(rg):
            array[k] = rg[j]
            j += 1
            k += 1
            
def Kfun(K, Org, row, col):
    """Extracts a K x K window from Org centered at (row, col) with zero-padding."""
    Krn = np.zeros((K, K, 3)) if Org.ndim == 3 else np.zeros((K, K))
    half_k = K // 2
    
    for i in range(K):
        for j in range(K):
            # Calculate source pixel coordinates in Org
            r = row - half_k + i
            c = col - half_k + j
            
            # Check image boundaries
            if 0 <= r < Org.shape[0] and 0 <= c < Org.shape[1]:
                Krn[i, j] = Org[r, c]
            # Out of bounds pixels remain 0 (zero padding)
            
    return Krn

def MediumOp(Ker, L):
    if Ker.ndim == 3:               # color image: median per channel
        V = Ker.reshape(-1, Ker.shape[-1])   # (K*K, channels)
        median = np.empty(Ker.shape[-1])
        for c in range(Ker.shape[-1]):
            col = V[:, c].copy()
            MS(col)
            median[c] = col[len(col)//2]
        return median
    V = Ker.reshape(-1)
    MS(V)                          # sort in place; MS returns nothing
    return V[len(V)//2]            # return the median (middle value)

def MediumFilter(Org, K=3):
    h, w = Org.shape[:2]
    Output = Org.copy()
    half_k = K // 2
    for x in range(h):
        for y in range(w):
            Ker = Kfun(K, Org, x, y)
            Output[x, y] = MediumOp(Ker, K)   # assign median to center
    return Output

            
def GaussOp(s,t,sigma):
    r=np.sqrt(s**2+t**2)
    return np.exp(-(r**2)/(2*(sigma**2)))
def SmoothBox(Org, K):
    """Applies box smoothing using kernel size K. SV is ignored; weights are 1/(K*K)."""
    h, w = Org.shape[:2]
    Output = Org.copy()  # Create a copy to prevent corrupting inputs during iteration
    scale = 1.0 / (K * K)          # normalized box kernel sum = 1
    
    for z in range(h):
        for p in range(w):
            Ker = Kfun(K, Org, z, p)
            # Apply normalized kernel scale and sum
            Output[z, p] = (Ker * scale).sum(axis=(0, 1)) if Org.ndim == 3 else (Ker * scale).sum()
            
    return np.clip(Output, 0, 255).astype(Org.dtype)

def GK(K, sigma):
    """Builds a K x K Gaussian Kernel using GaussOp."""
    half_k = K // 2
    kernel = np.zeros((K, K))
    
    for i in range(K):
        for j in range(K):
            # Convert matrix indices (i, j) to relative center coordinates (s, t)
            s = i - half_k
            t = j - half_k
            kernel[i, j] = GaussOp(s, t, sigma)
            
    # Normalize so all elements sum to 1 (prevents darkening/brightening)
    return kernel / kernel.sum()
def GaussSmooth(Org, K, sigma):
    """Applies Gaussian smoothing using kernel size K and standard deviation sigma."""
    h, w = Org.shape[:2]
    Output = Org.copy().astype(np.float32)
    
    # Generate the KxK matrix of Gaussian weights
    G_kernel = GK(K, sigma)
    
    for z in range(h):
        for p in range(w):
            Ker = Kfun(K, Org, z, p)
            
            if Org.ndim == 3:
                # Multiply KxKx3 pixel neighborhood by KxK kernel across RGB channels
                Output[z, p] = (Ker * G_kernel[:, :, np.newaxis]).sum(axis=(0, 1))
            else:
                Output[z, p] = (Ker * G_kernel).sum()
                
    return np.clip(Output, 0, 255).astype(Org.dtype)

#def SmoothBox(Org, K):
#    """Applies box smoothing using kernel size K and scale SV."""
#    h, w = Org.shape[:2]
#    Output = Org.copy()  # Create a copy to prevent corrupting inputs during iteration
    
#    for z in range(h):
#        for p in range(w):
#            Ker = Kfun(K, Org, z, p)
            # Apply kernel scale vector/value and sum
            
            
#    return Output

def LaplaceBorders(img, Klist):
    """
    Applies a list of 3x3 Laplacian kernels (Klist) to an image.
    Returns a list of edge-detected output images.
    """
    h, w = img.shape[:2]
    
    # Prepare an output image copy for each kernel in Klist
    outputs = [img.copy().astype(np.float32) for _ in Klist]
    
    # Loop through each pixel in the image
    for z in range(h):
        for p in range(w):
            # Extract 3x3 neighborhood window
            Ker = Kfun(3, img, z, p)
            
            # Apply each Laplacian kernel from Klist to the extracted window
            for idx, kernel in enumerate(Klist):
                if img.ndim == 3:
                    # Expand kernel dimensions to match 3D RGB array (3, 3, 1)
                    val = (Ker * kernel[:, :, np.newaxis]).sum(axis=(0, 1))
                else:
                    val = (Ker * kernel).sum()
                    
                outputs[idx][z, p] = val
                
    # Clip values to valid image intensity range [0, 255]
    return [np.clip(out, 0, 255).astype(img.dtype) for out in outputs]
def LaplaceBorder(img, Kerner):
    """Applies a single 3x3 Laplacian kernel to an image."""
    h, w = img.shape[:2]
    output = img.copy().astype(np.float32)
    
    # Ensure kernel has float data type for correct multiplication
    Kerner = np.array(Kerner, dtype=np.float32)
    
    for x in range(h):
        for y in range(w):
            Ker = Kfun(3, img, x, y)
            
            if img.ndim == 3:
                # Multiply 3x3x3 window by 3x3 kernel and sum height/width dimensions
                output[x, y] = (Ker * Kerner[:, :, np.newaxis]).sum(axis=(0, 1))
            else:
                # For grayscale images
                output[x, y] = (Ker * Kerner).sum()
                
    # Clip values to standard image intensity bounds [0, 255]
    return np.clip(output, 0, 255).astype(img.dtype)



SobelY=np.array([
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]
],dtype=np.float32)

SobelX=np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
],dtype=np.float32)
def Neg(img):
    Neg_img=255-img
    return Neg_img
#Smooth, Negative and then Laplace Border
def SgNL(img,K,Ktype,Sgm):
    SmoothImage=GaussSmooth(img,K,Sgm)
    Negative=Neg(SmoothImage)
    LaplaceImg=LaplaceBorder(Negative,Ktype)
    return LaplaceImg
print("SgNL Function")
def Sobel(img,VorH=True):
    if VorH is True:
        Kerner=SobelY
    else:
        Kerner=SobelX
    h, w = img.shape[:2]
    output = img.copy().astype(np.float32)
    
    for x in range(h):
        for y in range(w):
            Ker=Kfun(3,img,x,y)
            if img.ndim==3:
                output[x, y] = (Ker * Kerner[:, :, np.newaxis]).sum(axis=(0, 1))
            else:
                output[x, y] = (Ker * Kerner).sum()
    return np.clip(output, 0, 255).astype(img.dtype)
            
def SobelMag(img):
    gx=Sobel(img,VorH=True).astype(np.float32)
    gy=Sobel(img,VorH=False).astype(np.float32)
    
    mag=np.sqrt(gx**2+gy**2)
    
    return np.clip(mag,0,255).astype(img.dtype)

