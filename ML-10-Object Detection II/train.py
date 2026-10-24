from ultralytics import YOLO
from pathlib import Path

def main():
    # โหลดพรีทรีนโมเดล YOLOv8 Nano
    model = YOLO("yolov8n.pt")

    # เริ่มเทรน
    model.train(
        data="dataset/data.yaml",  # ชี้ไปที่ไฟล์ config ที่เราเพิ่งแก้
        epochs=25,                  # จำนวนรอบในการเทรน
        imgsz=640,                  # ขนาดภาพ
        batch=8,                    # ขนาด Batch (ปรับลดหรือเพิ่มตาม VRAM การ์ดจอ)
        device='cpu',                   # ใช้ GPU
        workers=0,
        amp=True,
        project="runs",
        name="aquarium_model",
        exist_ok=True,
        save=True,
        plots=True
    )

    save_dir = Path(model.trainer.save_dir)
    print("\nTraining completed!")
    print(f"Best Model: {save_dir / 'weights/best.pt'}")

if __name__ == "__main__":
    main()