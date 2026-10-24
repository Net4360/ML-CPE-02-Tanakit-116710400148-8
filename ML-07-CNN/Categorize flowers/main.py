import os
from data_loader import load_data
from split_data import split_dataset
from cnn_model import create_config_1, create_config_2
from evaluate import evaluate_model, plot_history, plot_confusion_matrix
from test_cnn import predict_and_visualize

def main():
    # สร้างโฟลเดอร์ outputs หากยังไม่มี
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

    print("\n--- 3. Build & Train CNN Models (Comparing Epochs & Configurations) ---")
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    num_classes = len(class_names)

    # Train Configuration 1 (5 Epochs)[cite: 1]
    print("\n[Training Config 1: 1 Conv Layer | 5 Epochs]")
    model_cfg1 = create_config_1(input_shape, num_classes)
    model_cfg1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_cfg1 = model_cfg1.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), verbose=1)

    # Train Configuration 2 (10 Epochs)[cite: 1]
    print("\n[Training Config 2: 2 Conv Layers + Dropout | 10 Epochs]")
    model_cfg2 = create_config_2(input_shape, num_classes)
    model_cfg2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_cfg2 = model_cfg2.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), verbose=1)

    print("\n--- 4. Evaluate Each Model Using Accuracy ---")
    loss_1, acc_1 = evaluate_model(model_cfg1, X_test, y_test)
    loss_2, acc_2 = evaluate_model(model_cfg2, X_test, y_test)

    print("\n================ FINAL PERFORMANCE COMPARISON ================")
    print(f"Config 1 (1 Conv Layer, 5 Epochs)  -> Test Accuracy: {acc_1:.4f}")
    print(f"Config 2 (2 Conv Layers, 10 Epochs) -> Test Accuracy: {acc_2:.4f}")
    print("==============================================================")

    # บันทึกผลลัพธ์ Text ลงใน outputs
    output_text = f"""=== ผลการทดลอง Lab 7: CNN ===
Dataset: Flowers Recognition
- Config 1 (1 Conv Layer, 5 Epochs) Test Accuracy: {acc_1:.4f} (Loss: {loss_1:.4f})
- Config 2 (2 Conv Layers, 10 Epochs) Test Accuracy: {acc_2:.4f} (Loss: {loss_2:.4f})
"""
    with open("outputs/lab7_output_results.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
    print("-> บันทึกผลการทดสอบลงไฟล์: outputs/lab7_output_results.txt เรียบร้อยแล้ว")

    print("\n--- 5. Saving Training History and Confusion Matrix ---")
    plot_history(history_cfg2, title_prefix="CNN Config 2", filename="outputs/training_history.png")
    plot_confusion_matrix(model_cfg2, X_test, y_test, class_names, filename="outputs/confusion_matrix.png")

    print("\n--- 6. Predictions on the Selected Dataset ---")
    predict_and_visualize(model_cfg2, X_test, y_test, class_names, filename="outputs/prediction_sample.png")

if __name__ == "__main__":
    main()