from pathlib import Path
import argparse
import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT = Path(__file__).resolve().parent / "outputs"
MODEL_PATH = OUT / "model.keras"
IMG_SIZE = 64

def preprocess(crop):
    return cv2.cvtColor(cv2.resize(crop, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--threshold', type=float, default=0.85)
    args = p.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดลที่: {MODEL_PATH} (กรุณารัน train.py ก่อน)")

    if not DATASET.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ dataset ที่: {DATASET}")

    # ดึงรายชื่อคลาสจากชื่อโฟลเดอร์จริงโดยอัตโนมัติ (เรียงตามตัวอักษร)
    class_names = sorted([d.name for d in DATASET.iterdir() if d.is_dir()])
    print(f"ตรวจพบรายชื่อคลาส: {class_names}")

    model = tf.keras.models.load_model(MODEL_PATH)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'ไม่สามารถเปิดกล้องเว็บแคมได้ (Camera ID: {args.camera})')

    print('กดปุ่ม Q บนคีย์บอร์ดเพื่อปิดโปรแกรม (Real-time Classifier Demo)')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            h, w = frame.shape[:2]
            # ครอปพื้นที่บริเวณตรงกลางของเฟรมกล้องมาทำนายผล
            x1, y1, x2, y2 = int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)
            crop_img = frame[y1:y2, x1:x2]
            
            if crop_img.size == 0:
                continue

            crop = preprocess(crop_img)
            pred = model.predict(np.expand_dims(crop, axis=0), verbose=0)[0]
            cls = int(np.argmax(pred))
            conf = float(np.max(pred))

            # หากค่าความมั่นใจผ่านเกณฑ์ threshold และคลาสถูกต้อง
            if conf >= args.threshold and cls < len(class_names):
                label_text = f'{class_names[cls]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Real-time Classifier Demo (Q to quit)', frame)
            
            # กดปุ่ม q เพื่อออกจากการทำงาน
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()