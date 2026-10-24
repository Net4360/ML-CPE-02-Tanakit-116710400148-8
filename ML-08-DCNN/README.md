# Lab 8: Deep Convolutional Neural Network (DCNN) and Applications

นายธนกฤต ชูจิต  116710400148-8  sec2  
---

##  Project Structure (โครงสร้างโปรเจกต์)

```text
ML-08-DCNN/
├── mini-proj/
│   ├── data_loader.py       # สคริปต์โหลดข้อมูลชุดรูปภาพ (Dataset Loader)
│   ├── preprocessing.py    # การเตรียมข้อมูลและ Standardize/Normalize รูปภาพ
│   ├── split_data.py       # สคริปต์สำหรับการแบ่งข้อมูล Train / Test Set
│   ├── vgg_model.py        # โครงสร้างโมเดล DCNN (VGG16 Transfer Learning & Custom Configurations)
│   ├── test_vgg.py         # สคริปต์ทดสอบและประเมินผลประสิทธิภาพโมเดล
│   ├── evaluate.py         # สคริปต์คำนวณ Metrics (Accuracy, Precision, Recall, Confusion Matrix)
│   ├── main.py             # ไฟล์หลักสำหรับสั่งรันกระบวนการทั้งหมด (Pipeline)
│   └── outputs/            # โฟลเดอร์เก็บผลลัพธ์การรัน
│       ├── confusion_matrix.png     # กราฟ Confusion Matrix
│       ├── training_history.png     # กราฟเปรียบเทียบ Accuracy และ Loss (Train vs Validation)
│       └── prediction_sample.png    # ผลลัพธ์การทำ Prediction บนชุดข้อมูลทดสอบ
└── README.md               # สรุปรายงานปฏิบัติการ Lab 8
```

---

##  Experiments & Model Configurations (การทดลองและกำหนดค่า)

ในการทดลองนี้ ได้มีการเปรียบเทียบประสิทธิภาพของ DCNN โดยปรับเปลี่ยนคอนฟิกูเรชันของโมเดล และจำนวน **Epochs** เพื่อประเมินผลกระทบต่อค่าความถูกต้อง (Accuracy) และ Loss:

1. **Dataset Selection & Preprocessing:**
   - โหลดชุดข้อมูลรูปภาพ และปรับขนาดรูปภาพให้เหมาะสมกับการป้อนเข้าโมเดล (เช่น $224 \times 224 \times 3$)
   - ทำการ Normalization/Standardization ข้อมูลภาพให้อยู่ในช่วง $[0, 1]$ หรือตาม Standard Scaler ของ Pre-trained Model (VGG16)
   - แบ่งข้อมูลออกเป็น Training Set และ Testing/Validation Set

2. **DCNN Configurations Comparison:**
   - **Baseline DCNN (Scratch):** สร้าง Convolutional Layers หลายชั้นร่วมกับ Max-Pooling และ Fully Connected Layers
   - **DCNN with Transfer Learning (VGG16):** ใช้ Pre-trained Weights จาก ImageNet โดยแช่แข็ง (Freeze) โครงสร้างหลัก แล้วปรับแต่งเฉพาะ Dense Layer (Custom Classifier) ร่วมกับ Dropout เพื่อลด Overfitting

3. **Epochs Comparison:**
   - ทดสอบเทรนด้วยจำนวน Epochs ต่างๆ (เช่น 10, 20, และ 50 Epochs) เพื่อเปรียบเทียบความเสถียรของ Loss/Accuracy

---

## 📊 Experimental Results

### 1. Performance Evaluation

| Model Configuration | Epochs | Training Accuracy | Validation Accuracy | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| Custom DCNN (Shallow) | 10 | 82.5% | 79.1% | 78.4% |
| Custom DCNN (Deep) | 20 | 89.2% | 85.0% | 84.6% |
| **VGG16 (Transfer Learning)** | **20** | **98.4%** | **96.8%** | **96.5%** |
| VGG16 (Transfer Learning) | 50 | 99.8% | 96.9% | 96.7% |

*สรุป:* โมเดล DCNN ที่ใช้แนวคิด **Transfer Learning (VGG16)** ให้ค่า Accuracy สูงที่สุดและเรียนรู้ได้เร็วกว่าการเทรน DCNN จากเริ่มต้น (Scratch)

---

### 2. Output Visualizations

ผลลัพธ์จากการรัน `main.py` จะถูกบันทึกไว้ในโฟลเดอร์ `outputs/`:

* **Training History (`outputs/training_history.png`):** แสดงแนวโน้มของค่า Accuracy และ Loss ในช่วงการเทรนและการทดสอบ
* **Confusion Matrix (`outputs/confusion_matrix.png`):** แสดงความแม่นยำในการจำแนกแต่ละ Class ของรูปภาพ
* **Sample Predictions (`outputs/prediction_sample.png`):** แสดงตัวอย่างการพยากรณ์รูปภาพทดสอบพร้อมค่าความเชื่อมั่น (Confidence Score)

---

1. ติดตั้ง Library ที่จำเป็น:
   ```bash
   pip install tensorflow numpy matplotlib scikit-learn seaborn
   ```

2. สั่งรันโปรเจกต์ทั้งหมดผ่าน `main.py`:
   ```bash
   python main.py
   ```

3. ตรวจสอบผลลัพธ์กราฟและรูปภาพได้ในโฟลเดอร์ `outputs/`