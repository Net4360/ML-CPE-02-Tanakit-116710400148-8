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
    print("--- กำลังเทรน 02-Fast-R-CNN (ROI Feature Pooling Approximation) ---")
    
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

    # สถาปัตยกรรมแบบ Fast R-CNN (จำลองการทำ Fixed-size ROI Feature Pooling ด้วย MaxPooling ขนาดใหญ่ก่อน Flatten)
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    
    # จำลอง ROI Pooling layer สำหรับปรับขนาด Feature ให้คงที่ก่อนเข้า Fully Connected
    x = layers.MaxPooling2D(pool_size=4, name='roi_feature_pooling')(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(len(class_names), activation='softmax', name='class_output')(x)

    model = tf.keras.Model(inputs, outputs, name='Fast_RCNN_Classifier')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    
    model.save(MODEL_PATH)
    print(f"Saved Fast R-CNN model to {MODEL_PATH}")

if __name__ == "__main__":
    main()