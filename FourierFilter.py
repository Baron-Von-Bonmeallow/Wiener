
import matplotlib.pyplot as plt
import numpy as np
import cv2

path = r"C:\Users\memip\Downloads\puppy.jpg"
pic = cv2.imread(path)
pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

plt.imshow(pic)
plt.axis("off")
plt.show()

def load_rgb(path):
    """Carga una imagen con OpenCV y la convierte a RGB."""
    pic = cv2.imread(path)
    if pic is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

def normalize_uint8(img):
    """Reescala a 0-255 preservando el contraste.

    Se usa cuando el resultado tiene valores negativos (p. ej. tras quitar la
    componente DC), donde un clip directo destruiria la mitad de la senal.
    """
    img = img - img.min()
    m = img.max()
    if m > 0:
        img = img / m
    return (img * 255).astype(np.uint8)


def Distance(shape):
    fl,cl=shape[:2]
    u0,v0=fl//2,cl//2
    u, v = np.indices((fl, cl))
    return np.sqrt((u - u0) ** 2 + (v - v0) ** 2)

def Spectrum(F):
    s = np.log1p(np.abs(F))
    return s.mean(axis=2) if s.ndim == 3 else s

def fft2c(image):
    F = np.fft.fft2(image.astype(np.float64), axes=(0, 1))
    return np.fft.fftshift(F, axes=(0, 1))

def ifft2c(Fshift):
    F = np.fft.ifftshift(Fshift, axes=(0, 1))
    return np.real(np.fft.ifft2(F, axes=(0, 1)))

def apply_filter(image, H):
    Fshift = fft2c(image)
    Gshift = Fshift * broadcast_filter(H, image)
    fltimg = ifft2c(Gshift)
    return fltimg, Spectrum(Fshift), Spectrum(Gshift), H
#---------------Image Printing

def broadcast_filter(H, image):
    return H[:, :, np.newaxis] if image.ndim == 3 else H

def clip_uint8(img):
    """Recorta a 0-255. Valido para pasa-bajas, que conservan el rango."""
    return np.clip(img, 0, 255).astype(np.uint8)

def SPnoise(image, amount=0.05, seed=None):
    noisy = image.copy()
    f, c = image.shape[:2]
    pix_num = int(amount * f * c)

    rng = np.random.default_rng(seed)

    noisy[rng.integers(0, f, pix_num), rng.integers(0, c, pix_num)] = 255  # sal
    noisy[rng.integers(0, f, pix_num), rng.integers(0, c, pix_num)] = 0    # pimienta

    return noisy
# ------------------------------- Filters------------------------------ #

#--------------Low Filters
def L_DC_comp(image):
    f, c = image.shape[:2]
    H = np.ones((f, c))
    H[f // 2, c // 2] = 1

    return apply_filter(image,H)

def L_IdealFilter(image, D0):
    D = Distance(image.shape)
    H = (D <= D0).astype(np.float64)

    return apply_filter(image,H)

def L_GaussFilter(image, sigma):
    D = Distance(image.shape)
    H = np.exp(-(D ** 2) / (2.0 * sigma ** 2))

    return apply_filter(image,H)

def L_Butterworth(image, D0, n):
    D = Distance(image.shape)
    H = 1.0 / (1.0 + (D / D0) ** (2 * n))

    return apply_filter(image,H)

def H_DC_comp(image):
    f, c = image.shape[:2]
    H = np.ones((f, c))
    H[f // 2, c // 2] = 0

    return apply_filter(image,H)
#------------High Filters
def H_IdealFilter(image, D0):
    D = Distance(image.shape)
    H = 1-(D <= D0).astype(np.float64)

    return apply_filter(image,H)

def H_GaussFilter(image, sigma):
    D = Distance(image.shape)
    H = 1-np.exp(-(D ** 2) / (2.0 * sigma ** 2))

    return apply_filter(image,H)

def H_Butterworth(image, D0, n):
    D = Distance(image.shape)
    H = 1-(1.0 / (1.0 + (D / D0) ** (2 * n)))

    return apply_filter(image,H)
#------------------------------------Pasa Bandas-----------------------------
def BP_IdealFilter(image,low,up):
    D = Distance(image.shape)
    H = ((D >= low) & (D <= up)).astype(np.float64)
    return apply_filter(image, H)
def BP_GaussFilter(image, D0, W):
    D = Distance(image.shape)
    H = np.exp(-(((D**2 - D0**2) / (D * W + 1e-5)) ** 2))
    return apply_filter(image, H)
def BP_Butterworth(image, D0, W, n):
    D = Distance(image.shape)
    denom = 1.0 + ((D * W) / (D**2 - D0**2 + 1e-5)) ** (2 * n)
    H = 1.0 - (1.0 / denom)
    return apply_filter(image, H)
#------------------------------------Rechazo de Bandas-----------------------
def BR_IdealFilter(image, low, up):
    D = Distance(image.shape)
    H = ~((D >= low) & (D <= up))
    return apply_filter(image, H.astype(np.float64))

def BR_GaussFilter(image, D0, W):
    D = Distance(image.shape)
    H = 1.0 - np.exp(-(((D**2 - D0**2) / (D * W + 1e-5)) ** 2))
    return apply_filter(image, H)

def BR_Butterworth(image, D0, W, n):
    D = Distance(image.shape)
    denom = 1.0 + ((D * W) / (D**2 - D0**2 + 1e-5)) ** (2 * n)
    H = 1.0 / denom
    return apply_filter(image, H)
#---------------------Webel---------------------#

#---------------------Results-------------------#
def show_result(original, fltimg, spectrum, spectrumF, H, titulo):
    fig, ax = plt.subplots(1, 5, figsize=(18, 4))

    ax[0].imshow(original, cmap="gray")
    ax[0].set_title("Original")

    ax[1].imshow(spectrum, cmap="gray")
    ax[1].set_title("Espectro original")

    ax[2].imshow(H, cmap="gray")
    ax[2].set_title("H(u, v)")

    ax[3].imshow(spectrumF, cmap="gray")
    ax[3].set_title("Espectro filtrado")

    ax[4].imshow(fltimg, cmap="gray")
    ax[4].set_title(titulo)

    for a in ax:
        a.axis("off")

    fig.tight_layout()
    plt.show()

def report(nombre, fltimg):
    """Imprime media y rango del resultado antes de normalizar."""
    print(f"{nombre:<28} media={np.mean(fltimg):>12.4e}  "
          f"rango=[{fltimg.min():.1f}, {fltimg.max():.1f}]")

if __name__ == "__main__":
    path = r"C:\Users\memip\Downloads\puppy.jpg"
    pic = load_rgb(path)

    plt.imshow(pic)
    plt.axis("off")
    plt.title("Imagen original")
    plt.show()
    print(f"Media original: {np.mean(pic):.4f}\n")

    # ---------------- PASA-BAJAS sobre la imagen con ruido -----------------
    # Los pasa-bajas suavizan: se prueban contra ruido sal y pimienta.
    noisy = SPnoise(pic, amount=0.05, seed=0)

    plt.imshow(noisy)
    plt.axis("off")
    plt.title("Ruido sal y pimienta (5%)")
    plt.show()

    print("--- PASA-BAJAS (salida en rango valido -> clip) ---")
    pasabajas = [
        ("Solo DC (imagen plana)",  L_DC_comp(pic)),
        ("Ideal D0=50",             L_IdealFilter(noisy, D0=50)),
        ("Gaussiano sigma=30",      L_GaussFilter(noisy, sigma=30)),
        ("Butterworth D0=50, n=2",  L_Butterworth(noisy, D0=50, n=2)),
    ]
    for nombre, (fltimg, spec, specF, H) in pasabajas:
        report(nombre, fltimg)
        show_result(noisy, clip_uint8(fltimg), spec, specF, H,
                    f"Pasa-bajas: {nombre}")

    # ---------------- PASA-ALTAS sobre la imagen limpia --------------------
    # Un pasa-altas amplifica el ruido, asi que se prueba sobre el original.
    print("\n--- PASA-ALTAS (media cero -> normalizar) ---")
    pasaaltas = [
        ("Sin componente DC",       H_DC_comp(pic)),
        ("Ideal D0=50",             H_IdealFilter(pic, D0=50)),
        ("Gaussiano sigma=30",      H_GaussFilter(pic, sigma=30)),
        ("Butterworth D0=50, n=2",  H_Butterworth(pic, D0=50, n=2)),
    ]
    for nombre, (fltimg, spec, specF, H) in pasaaltas:
        report(nombre, fltimg)
        show_result(pic, normalize_uint8(fltimg), spec, specF, H,
                    f"Pasa-altas: {nombre}")

    # ---------------- PASA BANDAS (Band-Pass) ------------------------------
    # Extrae solo un rango específico de frecuencias (media cero -> normalizar)
    print("\n--- PASA BANDAS (media cercana a cero -> normalizar) ---")
    pasabandas = [
        ("Ideal low=20, up=60",      BP_IdealFilter(pic, low=20, up=60)),
        ("Gaussiano D0=40, W=20",    BP_GaussFilter(pic, D0=40, W=20)),
        ("Butterworth D0=40, W=20",  BP_Butterworth(pic, D0=40, W=20, n=2)),
    ]
    for nombre, (fltimg, spec, specF, H) in pasabandas:
        report(nombre, fltimg)
        show_result(pic, normalize_uint8(fltimg), spec, specF, H,
                    f"Pasa-bandas: {nombre}")

    # ---------------- RECHAZO DE BANDAS (Band-Reject) ----------------------
    # Elimina un rango específico de frecuencias conservando la base (clip)
    print("\n--- RECHAZO DE BANDAS (salida en rango valido -> clip) ---")
    rechazobandas = [
        ("Ideal low=20, up=60",      BR_IdealFilter(pic, low=20, up=60)),
        ("Gaussiano D0=40, W=20",    BR_GaussFilter(pic, D0=40, W=20)),
        ("Butterworth D0=40, W=20",  BR_Butterworth(pic, D0=40, W=20, n=2)),
    ]
    for nombre, (fltimg, spec, specF, H) in rechazobandas:
        report(nombre, fltimg)
        show_result(pic, clip_uint8(fltimg), spec, specF, H,
                    f"Rechazo de bandas: {nombre}")