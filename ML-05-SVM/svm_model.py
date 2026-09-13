from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def train_svm(X_train, y_train, kernel_type="rbf", pca_components=150):
    # [สำคัญ] ใช้ Pipeline มัดรวม StandardScaler และ PCA เข้าด้วยกัน
    # เพื่อป้องกันปัญหา Data Leakage (ข้อมูลทดสอบหลุดไปปนกับข้อมูลฝึกสอน)
    scaler = Pipeline([
        # 1. ปรับสเกลข้อมูลให้มีค่าเฉลี่ยเป็น 0 (ตรงตามโจทย์ Lab: Standardize the input)
        ("scaler", StandardScaler()),
        # 2. ทำ PCA บีบอัดข้อมูล คัดมาเฉพาะ 150 จุดเด่นที่สำคัญที่สุด ช่วยให้เทรนเร็วขึ้น
        ("pca", PCA(n_components=min(pca_components, len(X_train)), whiten=True, random_state=42)),
    ])
    
    # แปลงข้อมูล X_train ด้วย Pipeline ที่ตั้งไว้
    X_train_scaled = scaler.fit_transform(X_train)
    
    # สร้างโมเดล SVM และกำหนดประเภทของ Kernel (Linear, Poly, RBF ตามที่รับค่ามา)
    model = SVC(kernel=kernel_type, C=10, gamma="scale")
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def predict_svm(model, scaler, X_test):
    # [สำคัญ] ข้อมูล Test ต้องนำมาผ่าน Scaler ตัวเดียวกับตอน Train เสมอ
    X_test_scaled = scaler.transform(X_test)
    return model.predict(X_test_scaled)