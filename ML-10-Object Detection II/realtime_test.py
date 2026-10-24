from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    weights = ROOT / 'runs' / 'detect' / 'runs' / 'aquarium_model' / 'weights' / 'best.pt'
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam')

    print("Starting webcam... Press 'q' to exit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(source=frame, conf=0.25, imgsz=640, device='cpu', verbose=False)
            annotated_frame = results[0].plot()

            cv2.imshow('Aquarium Real-time Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()