from pathlib import Path
import random
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    weights = ROOT / 'runs' / 'detect' / 'runs' / 'aquarium_model' / 'weights' / 'best.pt'
    image_dir = ROOT / 'dataset' / 'test' / 'images'

    if not weights.exists() or not image_dir.exists():
        raise FileNotFoundError('Model or image directory not found!')

    model = YOLO(str(weights))
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    selected_images = random.sample(images, min(5, len(images)))

    for image in selected_images:
        results = model.predict(
            source=str(image),
            conf=0.25,
            imgsz=640,
            device='cpu',
            save=True,
            project=str(ROOT / 'runs'),
            name='aquarium_image_test',
            exist_ok=True
        )

        for result in results:
            print(f'\nImage: {image.name}')
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                print(f' - {model.names[class_id]}: {confidence:.2f}')

            annotated = result.plot()
            cv2.imshow('Aquarium Detection Test', annotated)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print('\nImage testing completed! Results saved in runs/aquarium_image_test')

if __name__ == '__main__':
    main()