# Earphone Classification ด้วย KNN (TensorFlow)

โปรเจกต์นี้เป็นการทดลองใช้ **K-Nearest Neighbors (KNN)** เพื่อจำแนกประเภทของหูฟังจากข้อมูลในไฟล์ `earphone.csv`

ประเภทที่ต้องการทำนายมี 3 กลุ่ม คือ

- `In the Ear`
- `On the Ear`
- `True Wireless`

## ข้อมูลที่ใช้

โปรแกรมใช้ข้อมูลตัวเลข 5 ค่าเป็น Features สำหรับการทำนาย ได้แก่

- `ratings` คะแนนสินค้า
- `reviews` จำนวนรีวิว
- `selling_price` ราคาขาย
- `actual_price` ราคาปกติ
- `off` เปอร์เซ็นต์ส่วนลด

ส่วนประเภทของหูฟังจะดึงมาจากคอลัมน์ `colours` แล้วนำมาใช้เป็น Target สำหรับการจำแนก

## ขั้นตอนการทำงาน

### 1. เตรียมข้อมูล (`data_loader.py`)

โปรแกรมอ่านข้อมูลจาก CSV แล้วทำความสะอาดข้อมูล เช่น

- ลบเครื่องหมาย `₹`, `,`, `% off` และวงเล็บออก
- แปลงข้อมูลที่เป็นข้อความให้เป็นตัวเลข
- แปลงชื่อประเภทหูฟังให้เป็นตัวเลขสำหรับใช้กับโมเดล
- แบ่งข้อมูลเป็น **Train 60% / Validation 20% / Test 20%**
- ใช้ `StandardScaler` เพื่อปรับสเกลข้อมูลก่อนนำไปใช้กับ KNN

### 2. สร้าง KNN (`knn_tf.py`)

สร้าง KNN ด้วย TensorFlow โดยหลักการคือ

1. คำนวณระยะห่างระหว่างข้อมูลใหม่กับข้อมูล Train ด้วย Euclidean Distance
2. เลือกข้อมูลที่อยู่ใกล้ที่สุดจำนวน `k` ตัว
3. ดูว่ากลุ่มเพื่อนบ้านส่วนใหญ่อยู่ Class ไหน
4. เลือก Class ที่มีคะแนนโหวตมากที่สุดเป็นคำตอบ

### 3. เลือกค่า k ที่ดีที่สุด (`main.py`)

ทดลองค่า `k` หลายค่า คือ

`1, 2, 3, 5, 10, 11, 15, 21, 30`

จาก Validation Set พบว่า **k = 11** ให้ผลดีที่สุด โดยมี Validation Accuracy ประมาณ **75.1%**

![กราฟการเลือกค่า k](01_k_curve.png)

### 4. ทดสอบโมเดล

หลังจากเลือก `k = 11` แล้ว นำโมเดลไปทดสอบกับ Test Set

ผลที่ได้คือ **Test Accuracy = 70.27%** หรือทำนายถูก **130 จาก 185 ตัวอย่าง**

Confusion Matrix ที่ได้คือ

| True / Predicted | In the Ear | On the Ear | True Wireless |
|---|---:|---:|---:|
| In the Ear | 95 | 1 | 8 |
| On the Ear | 15 | 5 | 5 |
| True Wireless | 22 | 4 | 30 |

จากตารางจะเห็นว่าโมเดลทำนาย `In the Ear` ถูก 95 ตัวอย่าง, `On the Ear` ถูก 5 ตัวอย่าง และ `True Wireless` ถูก 30 ตัวอย่าง

![Confusion Matrix](02_confusion_matrix.png)

### 5. ตรวจสอบผลลัพธ์

โปรแกรมมีการเปรียบเทียบ KNN ที่เขียนด้วย TensorFlow กับ `KNeighborsClassifier` ของ scikit-learn เพื่อดูว่าการทำงานให้ผลใกล้เคียงกันหรือไม่

นอกจากนี้ยังเปรียบเทียบกับ Baseline ที่ใช้วิธีทาย Class ที่พบมากที่สุดทุกครั้ง เพื่อดูว่าโมเดลสามารถเรียนรู้จาก Features ได้มากน้อยเพียงใด

## ไฟล์สำคัญในโปรเจกต์

- `data_loader.py` : อ่านข้อมูล, ทำความสะอาด, แบ่งชุดข้อมูล และ Scaling
- `knn_tf.py` : KNN ที่เขียนด้วย TensorFlow
- `evaluate.py` : สร้างกราฟ, Confusion Matrix และบันทึกผลการทำนาย
- `main.py` : โปรแกรมหลักสำหรับหา k ที่ดีที่สุดและทดสอบโมเดล
- `run_all.py` : ใช้รันไฟล์ Python ในโปรเจกต์
- `predictions.csv` : ผลการทำนายของ Test Set
- `01_k_curve.png` : กราฟ Validation Accuracy ของแต่ละค่า k
- `02_confusion_matrix.png` : Confusion Matrix

## วิธีรันโปรแกรม

ติดตั้ง Library ที่จำเป็นก่อน

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

จากนั้นรัน

```bash
python main.py
```

## สรุป

โปรเจกต์นี้แสดงขั้นตอนการสร้าง KNN ตั้งแต่การเตรียมข้อมูล การเลือกค่า `k` ด้วย Validation Set ไปจนถึงการทดสอบด้วย Test Set โดยค่า **k = 11** ให้ผลดีที่สุดในชุด Validation และโมเดลได้ Accuracy บน Test Set ประมาณ **70.27%**
