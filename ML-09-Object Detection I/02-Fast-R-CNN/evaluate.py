import tensorflow as tf
from tensorflow.keras import models
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import json

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT / "model.keras"

IMG_SIZE = 64
BATCH_SIZE = 32

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดลที่: {MODEL_PATH} (กรุณารัน train.py ก่อน)")

    if not DATASET.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ dataset ที่: {DATASET}")

    print("กำลังโหลดชุดข้อมูลสำหรับประเมินผล (Validation/Test Split)...")
    
    # โหลดข้อมูลชุดทดสอบ/ตรวจสอบโดยใช้โครงสร้างโฟลเดอร์คลาสเดิม
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = val_ds.class_names
    print(f"รายชื่อคลาส: {class_names}")

    print(f"กำลังโหลดโมเดลจาก: {MODEL_PATH}")
    model = models.load_model(MODEL_PATH)

    y_true = []
    y_pred = []

    print("กำลังประเมินผลโมเดล...")
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nผลการประเมิน (Evaluation Results)")
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    metrics = {
        'accuracy': float(acc)
    }
    
    eval_json_path = OUT / 'evaluation.json'
    eval_json_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f"บันทึกผลการประเมินลงที่: {eval_json_path}")

if __name__ == "__main__":
    main()