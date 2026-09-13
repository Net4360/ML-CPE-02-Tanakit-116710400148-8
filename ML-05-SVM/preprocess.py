import cv2
import numpy as np

def preprocess_image(image, img_size=100):
    if image is None or image.size == 0:
        return None
        
    # [สำคัญ] แปลงภาพสีเป็นขาวดำ (Grayscale) 
    # เพื่อให้โมเดลโฟกัสแค่ "รูปร่างมือ" โดยไม่เอาสีผิวหรือฉากหลังมาเป็นปัจจัย
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # ย่อขนาดรูปภาพเป็น 100x100 เพื่อลดภาระการคำนวณของเครื่อง
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return image

def to_features(images):
    # [สำคัญ] แปลงภาพ 2 มิติ (ตารางพิกเซล) ให้เป็นเส้นตรง 1 มิติ (Flatten)
    # เพราะโมเดล SVM รับข้อมูลรูปแบบ 1D Array เท่านั้น
    features = images.reshape(len(images), -1).astype(np.float32)
    
    # ปรับสเกลค่าพิกเซลจาก 0-255 ให้อยู่ในช่วง 0-1 (Normalization พื้นฐาน)
    features /= 255.0
    return features