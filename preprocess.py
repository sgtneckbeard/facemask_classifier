import numpy as np
import cv2
from PIL import Image

def preprocess_image(image: Image.Image):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    brightness_beta = 13
    contrast_alpha = 1.2
    hi = np.percentile(img_array, 99)
    img_array = np.clip(img_array, 0, hi)
    img_array = cv2.convertScaleAbs(img_array, alpha=contrast_alpha, beta=brightness_beta)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_array = clahe.apply(np.uint8(img_array))
    img_array = cv2.resize(img_array, (300, 224))
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 224, 300, 1)
    return img_array