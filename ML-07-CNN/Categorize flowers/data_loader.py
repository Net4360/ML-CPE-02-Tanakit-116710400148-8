import os
import cv2
import numpy as np
from preprocessing import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def load_data(data_path="flowers", img_size=128, max_per_class=None):
    """Select and load a dataset of your choice[cite: 1]"""
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {os.path.abspath(data_path)}\n"
            "กรุณาสร้างโฟลเดอร์ 'flowers' และใส่ซับโฟลเดอร์ย่อยตามชื่อคลาส (เช่น daisy, rose ฯลฯ)"
        )

    images = []
    labels = []

    classes = sorted([
        folder for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder))
    ])
    print("Detected classes:", classes)

    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        filenames = sorted([
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        ])

        loaded = 0
        skipped = 0
        for filename in filenames:
            if max_per_class and loaded >= max_per_class:
                break

            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)
            
            processed_image = preprocess_image(image, img_size)

            if processed_image is None:
                skipped += 1
                continue

            images.append(processed_image)
            labels.append(label)
            loaded += 1

        print(f"Loaded class {class_name}: {loaded} images ({skipped} skipped)")

    return np.array(images), np.array(labels), classes