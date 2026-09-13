import os
import cv2
import numpy as np
from preprocess import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def load_data(data_path, img_size=100, max_per_class=None):
    images, labels = [], []
    
    # อ่านชื่อโฟลเดอร์ย่อย (rock, paper, scissors) เพื่อนำมาตั้งเป็นชื่อคลาสอัตโนมัติ
    classes = sorted([f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))])
    print("Detected classes:", classes)
    
    # ลูปเข้าไปในแต่ละโฟลเดอร์เพื่อดึงรูปภาพออกมา
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        filenames = sorted(f for f in os.listdir(class_path) if f.lower().endswith(VALID_EXT))
        
        loaded = 0
        for filename in filenames:
            # ใช้จำกัดจำนวนรูปภาพเพื่อไม่ให้คอมพิวเตอร์ทำงานหนักเกินไปตอนพรีเซนต์
            if max_per_class and loaded >= max_per_class: break
            
            image = cv2.imread(os.path.join(class_path, filename))
            image = preprocess_image(image, img_size)
            
            if image is not None:
                images.append(image)
                labels.append(label) # เก็บตัวเลข Label ให้ตรงกับรูปภาพ
                loaded += 1
                
    # คืนค่ากลับไปเป็น Numpy Array เพื่อให้พร้อมคำนวณทางคณิตศาสตร์
    return np.stack(images), np.array(labels), classes