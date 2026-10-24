import tensorflow as tf
from tensorflow.keras import models
from pathlib import Path
import numpy as np
import cv2
import random

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT / "model.keras"

IMG_SIZE = 64

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดลที่: {MODEL_PATH} (กรุณารัน train.py ก่อน)")

    if not DATASET.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ dataset ที่: {DATASET}")

    # ค้นหาภาพทั้งหมดจากทุกโฟลเดอร์คลาสข้างใน dataset
    image_extensions = (".jpg", ".jpeg", ".png", ".webp")
    image_files = []
    
    for class_dir in DATASET.iterdir():
        if class_dir.is_dir():
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    image_files.append((img_file, class_dir.name))

    if not image_files:
        raise RuntimeError(f"ไม่พบไฟล์รูปภาพในโฟลเดอร์คลาสย่อยของ {DATASET}")

    # สุ่มเลือกรูปภาพมา 1 รูปสำหรับการทดสอบ
    img_path, true_class = random.choice(image_files)
    print(f"สุ่มเลือกรูปภาพทดสอบจากคลาส '{true_class}': {img_path.name}")

    # อ่านภาพด้วย OpenCV
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"ไม่สามารถอ่านไฟล์ภาพได้: {img_path}")

    # ปรับขนาดภาพให้เท่ากับตอนที่นำไปเทรน
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)

    # โหลดโมเดลมาทำนายผล
    print(f"กำลังโหลดโมเดลจาก: {MODEL_PATH}")
    model = models.load_model(MODEL_PATH)
    
    # ดึงรายชื่อคลาสจากโฟลเดอร์อัตโนมัติ (เรียงตามตัวอักษรเหมือนตอนใช้ image_dataset_from_directory)
    class_names = sorted([d.name for d in DATASET.iterdir() if d.is_dir()])
    
    predictions = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    pred_class = class_names[pred_idx]

    print(f"ผลการทำนาย: {pred_class} (ความมั่นใจ: {confidence * 100:.2f}%)")
    print(f"คลาสที่แท้จริง (True Class): {true_class}")

    # วาดข้อความผลลัพธ์ลงบนภาพต้นฉบับ
    h, w, _ = image.shape
    text = f"Pred: {pred_class} ({confidence:.2f})"
    color = (0, 255, 0) if pred_class == true_class else (0, 0, 255)
    
    cv2.putText(image, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(image, f"True: {true_class}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # บันทึกภาพผลลัพธ์
    save_path = OUT / "image_result.jpg"
    cv2.imwrite(str(save_path), image)
    print(f"บันทึกภาพผลลัพธ์ลงที่: {save_path}")

if __name__ == "__main__":
    main()