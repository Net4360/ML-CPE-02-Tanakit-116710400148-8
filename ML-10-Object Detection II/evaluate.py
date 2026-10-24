from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    # ชี้ไปยังพาร์ทโมเดลที่เทรนเสร็จตามผลลัพธ์ใน Log ของคุณ
    weights = ROOT / 'runs' / 'detect' / 'runs' / 'aquarium_model' / 'weights' / 'best.pt'

    if not weights.exists():
        raise FileNotFoundError(f'Model not found: {weights}')

    model = YOLO(str(weights))

    # ประเมินผลบน Test dataset
    metrics = model.val(
        data=str(ROOT / 'dataset' / 'data.yaml'),
        split='test',
        imgsz=640,
        batch=8,
        device='cpu',
        plots=True,
        project=str(ROOT / 'runs'),
        name='aquarium_evaluation',
        exist_ok=True,
    )

    print('\n===== Final Evaluation Results =====')
    print(f'Precision: {metrics.box.mp:.4f}')
    print(f'Recall: {metrics.box.mr:.4f}')
    print(f'mAP50: {metrics.box.map50:.4f}')
    print(f'mAP50-95: {metrics.box.map:.4f}')

if __name__ == '__main__':
    main()