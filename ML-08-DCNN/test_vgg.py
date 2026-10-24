import os
import numpy as np
import matplotlib.pyplot as plt

def predict_and_visualize(model, X_test, y_test, class_names, filename="outputs/prediction_sample.png"):
    """สุ่มทำนายผลและบันทึกภาพ Prediction Sample"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    predictions = model.predict(X_test)
    num_samples = min(5, len(X_test))

    plt.figure(figsize=(12, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        
        # แสดงภาพที่สเกลกลับให้อยู่ในช่วงภาพปกติ
        img_disp = X_test[i].astype('uint8')
        plt.imshow(img_disp)
        
        predicted_label = np.argmax(predictions[i])
        true_label = y_test[i]

        color = 'blue' if predicted_label == true_label else 'red'
        true_name = class_names[true_label] if not isinstance(class_names, np.ndarray) else class_names[true_label]
        pred_name = class_names[predicted_label]
        plt.xlabel(f"Pred: {pred_name}\nTrue: {true_name}", color=color)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"-> Saved prediction sample image: {filename}")
    plt.close()