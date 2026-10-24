import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT / "model.keras"

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 5

def main():
    print("--- กำลังเทรน 01-R-CNN (Standard Region Classifier) ---")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET, validation_split=0.2, subset="training", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    # สถาปัตยกรรมแบบ R-CNN (Sequential มาตรฐานสำหรับสกัดฟีเจอร์และจำแนกคลาส)
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names), activation='softmax', name='class_output')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    
    model.save(MODEL_PATH)
    print(f"Saved R-CNN model to {MODEL_PATH}")

if __name__ == "__main__":
    main()