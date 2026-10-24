import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(image, img_size=128):
    """ปรับขนาดรูปภาพ และเตรียมพิกเซลตามข้อกำหนดของ VGG16 Preprocessing"""
    if image is None:
        return None
    image = cv2.resize(image, (img_size, img_size))
    # แปลงสีจาก BGR เป็น RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    return image