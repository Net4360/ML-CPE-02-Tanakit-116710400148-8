import cv2
import numpy as np

def preprocess_image(image, img_size=128):
    """ปรับขนาดภาพและทำ Standardization พิกเซลให้อยู่ในช่วง [0, 1]"""
    if image is None:
        return None
    image = cv2.resize(image, (img_size, img_size))
    # Standardize input features[cite: 1]
    image = image.astype(np.float32) / 255.0
    return image