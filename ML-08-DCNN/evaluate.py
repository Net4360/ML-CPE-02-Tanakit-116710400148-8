import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_model(model, X_test, y_test):
    """ประเมินผลความแม่นยำและ Loss ของโมเดล"""
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return test_loss, test_acc

def plot_history(history, title_prefix="DCNN Model", filename="outputs/training_history.png"):
    """บันทึกกราฟแสดงผล Training และ Validation Accuracy / Loss"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{title_prefix} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title_prefix} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"-> Saved training history graph: {filename}")
    plt.close()

def plot_confusion_matrix(model, X_test, y_test, class_names, filename="outputs/confusion_matrix.png"):
    """สร้างและบันทึกรูปภาพ Confusion Matrix"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"-> Saved confusion matrix plot: {filename}")
    plt.close()