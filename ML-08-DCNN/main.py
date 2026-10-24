import os
import tensorflow as tf
from data_loader import load_data
from split_data import split_dataset
from vgg_model import create_vgg_config_1, create_vgg_config_2
from evaluate import evaluate_model, plot_history, plot_confusion_matrix
from test_vgg import predict_and_visualize

def main():
    os.makedirs("outputs", exist_ok=True)

    print("--- 1. Select and Load Dataset ---")
    DATA_PATH = "flowers"
    IMG_SIZE = 128

    try:
        X, y, class_names = load_data(DATA_PATH, img_size=IMG_SIZE, max_per_class=250)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Total images loaded: {X.shape[0]}, Classes: {class_names}")

    print("\n--- 2. Split Dataset into Training, Validation, and Testing ---")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y)

    print("\n--- 3. Build & Train DCNN Models (Comparing Epochs & Configurations) ---")
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    num_classes = len(class_names)

    # Train Configuration 1 (5 Epochs)
    print("\n[Training DCNN Config 1: VGG16 Base | 5 Epochs]")
    model_cfg1 = create_vgg_config_1(input_shape, num_classes)
    model_cfg1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_cfg1 = model_cfg1.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), verbose=1)

    # Train Configuration 2 (10 Epochs)
    print("\n[Training DCNN Config 2: Deep VGG16 + Dropout | 10 Epochs]")
    model_cfg2 = create_vgg_config_2(input_shape, num_classes)
    model_cfg2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_cfg2 = model_cfg2.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), verbose=1)

    print("\n--- 4. Evaluate Each Model Using Accuracy ---")
    loss_1, acc_1 = evaluate_model(model_cfg1, X_test, y_test)
    loss_2, acc_2 = evaluate_model(model_cfg2, X_test, y_test)

    print("\n================ FINAL PERFORMANCE COMPARISON ================")
    print(f"Config 1 (VGG16 Base, 5 Epochs)  -> Test Accuracy: {acc_1:.4f}")
    print(f"Config 2 (Deep VGG16, 10 Epochs) -> Test Accuracy: {acc_2:.4f}")
    print("==============================================================")

    print("\n--- 5. Saving Outputs ---")
    plot_history(history_cfg2, title_prefix="DCNN VGG16 Config 2", filename="outputs/training_history.png")
    plot_confusion_matrix(model_cfg2, X_test, y_test, class_names, filename="outputs/confusion_matrix.png")

    print("\n--- 6. Predictions on Selected Dataset ---")
    predict_and_visualize(model_cfg2, X_test, y_test, class_names, filename="outputs/prediction_sample.png")

if __name__ == "__main__":
    main()