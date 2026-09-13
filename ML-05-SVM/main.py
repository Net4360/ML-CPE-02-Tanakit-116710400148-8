import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_load import load_data
from preprocess import to_features
from svm_model import train_svm, predict_svm

DATA_PATH = "rps_data"
OUTPUT_DIR = "outputs"
IMG_SIZE = 100
MAX_PER_CLASS = 1000  

# ฟังก์ชันวาดกราฟตารางความสับสน (Confusion Matrix)
def plot_confusion_matrix(matrix, classes, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(classes))), ax.set_xticklabels(classes)
    ax.set_yticks(np.arange(len(classes))), ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"), ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    
    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, matrix[i, j], ha="center", va="center", color=color)
            
    fig.tight_layout()
    fig.savefig(save_path, dpi=150) # เซฟกราฟเป็นไฟล์ .png
    plt.close(fig)

# ฟังก์ชันสุ่มรูปภาพ 4 รูปมาทดสอบโมเดล แล้วเซฟภาพสรุปผล
def save_prediction_sample(model, scaler, X_test_img, X_test_feat, y_test, classes, save_path):
    indices = np.random.choice(len(X_test_feat), 4, replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        pred = predict_svm(model, scaler, [X_test_feat[idx]])[0]
        true_label = y_test[idx]
        color = "green" if pred == true_label else "red" # ทายถูกตัวหนังสือสีเขียว ผิดสีแดง
        
        axes[i].imshow(X_test_img[idx], cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Pred: {classes[pred]}\nTrue: {classes[true_label]}", color=color)
        
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def main():
    # สร้างโฟลเดอร์ outputs ไว้เก็บไฟล์ (ถ้ายังไม่มี)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("[1] Loading Data...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)
    features = to_features(images)
    
    # [Output Requirement] เซฟข้อมูลดิบเก็บไว้ (จำลองการสำรองข้อมูล)
    np.save(f"{OUTPUT_DIR}/images.npy", images)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("[2] Splitting Data...")
    # แบ่งข้อมูลเป็น Train 80% และ Test 20%
    X_train, X_test, y_train, y_test, _, X_test_img = train_test_split(
        features, labels, images, test_size=0.2, random_state=42, stratify=labels
    )
    
    # [Output Requirement] เซฟชุดข้อมูลที่แบ่งแล้วเก็บไว้
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print("[3] Training and Evaluating...")
    # [สำคัญ] เทรนโมเดล 3 แบบตามเงื่อนไขของโจทย์ Lab
    kernels = ["linear", "poly", "rbf"]
    best_model, best_scaler = None, None
    
    for k in kernels:
        print(f"\n--- Kernel: {k.upper()} ---")
        model, scaler = train_svm(X_train, y_train, kernel_type=k)
        
        # [Output Requirement] เซฟไฟล์โมเดล (.pkl) เอาไปใช้ต่อโดยไม่ต้องเทรนใหม่
        joblib.dump(model, f"{OUTPUT_DIR}/svm_model_{k}.pkl")
        joblib.dump(scaler, f"{OUTPUT_DIR}/scaler_{k}.pkl")
        
        # นำโมเดลมาทำข้อสอบ (Predict) และวัดคะแนน (Accuracy)
        predictions = predict_svm(model, scaler, X_test)
        print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
        
        # สร้างกราฟ Confusion Matrix ของแต่ละ Kernel
        cm = confusion_matrix(y_test, predictions)
        plot_confusion_matrix(cm, classes, f"{OUTPUT_DIR}/confusion_matrix_{k}.png")
        
        # เก็บโมเดล RBF ไว้ใช้แสดงภาพตัวอย่างตอนจบ (เพราะมักจะแม่นยำที่สุด)
        if k == "rbf":
            best_model, best_scaler = model, scaler

    print(f"\n[4] Saving prediction sample to {OUTPUT_DIR}/prediction_sample.png")
    # สร้างรูปสุ่มทายผล 4 รูป แล้วเซฟลงโฟลเดอร์
    save_prediction_sample(best_model, best_scaler, X_test_img, X_test, y_test, classes, f"{OUTPUT_DIR}/prediction_sample.png")
    print("Done! Check the 'outputs' folder.")

if __name__ == "__main__":
    main()