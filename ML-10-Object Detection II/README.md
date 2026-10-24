# ML-10 Object Detection II: Aquarium / Fish Detection with YOLOv8
โมเดล YOLOv8 บนชุดข้อมูล Aquarium / Fish Detection เพื่อตรวจจับและระบุตำแหน่งของสิ่งมีชีวิตและวัตถุต่างๆ ภายในตู้เลี้ยงสัตว์น้ำแบบเรียลไทม์
นายธนกฤต ชูจิต รหัสนักศึกษา116710400148-8 sec2
---

---

## รายละเอียดโปรเจกต์ (Project Overview)

การตรวจจับวัตถุในตู้เลี้ยงสัตว์น้ำมีความท้าทายจากการสะท้อนของแสง ความขุ่นของน้ำ วัตถุซ้อนทับกัน และขนาดที่หลากหลาย โปรเจกต์นี้เลือกใช้ YOLOv8 Nano (`yolov8n.pt`) ซึ่งเป็น Deep Learning ประเภท Single-stage Detector ที่ประมวลผลได้รวดเร็ว เหมาะกับการตรวจจับแบบ Real-time และสามารถทำงานบน CPU ได้อย่างมีประสิทธิภาพ

---

## โครงสร้างของโปรเจกต์ (Project Structure)

```text
ML-10-Object Detection II/
├── dataset/                    # โฟลเดอร์เก็บชุดข้อมูล (train, valid, test)
│   └── data.yaml               # ไฟล์คอนฟิกูเรชันกำหนดเส้นทางและคลาส
├── runs/                       # โฟลเดอร์เก็บผลลัพธ์การ Train, Evaluate และ Predict
├── train.py                    # สคริปต์สำหรับการฝึกสอนโมเดล
├── evaluate.py                 # สคริปต์สำหรับการประเมินวัดผล
├── image_test.py               # สคริปต์ทดสอบตรวจจับกับภาพนิ่ง
├── realtime_test.py            # สคริปต์ทดสอบตรวจจับ Real-time ผ่านเว็บแคม
└── README.md                   # เอกสารอธิบายโปรเจกต์
```

---

## รายละเอียดชุดข้อมูล (Dataset Details)

ชุดข้อมูล Aquarium Dataset อยู่ในรูปแบบ YOLO Format ประกอบด้วย 7 คลาส (`nc: 7`):

| Class ID | Class Name | คำอธิบายภาษาไทย |
| :---: | :--- | :--- |
| **0** | `fish` | ปลาทั่วไป |
| **1** | `jellyfish` | แมงกะพรุน |
| **2** | `penguin` | นกเพนกวิน |
| **3** | `puffin` | นกพัฟฟิน |
| **4** | `shark` | ปลาฉลาม |
| **5** | `starfish` | ปลาดาว |
| **6** | `stingray` | ปลากระเบน |

---

## ข้อกำหนดการติดตั้ง (Installation & Prerequisites)

### 1. เครื่องมือที่จำเป็น
* Python 3.9 ขึ้นไป
* OpenCV (`opencv-python`)
* Ultralytics YOLOv8 (`ultralytics`)
* PyTorch (`torch`, `torchvision`)

### 2. คำสั่งติดตั้ง
```bash
py -m pip install ultralytics opencv-python
```

---

## การทำงานของสคริปต์หลัก (Scripts Overview)

### 1. สคริปต์ฝึกสอนโมเดล (`train.py`)
* **หน้าที่:** โหลด Weights ตั้งต้น (`yolov8n.pt`) และทำการ Fine-tuning กับชุดข้อมูลใน `dataset/data.yaml`
* **ความจำเป็น:** เป็นขั้นตอนสำคัญในการสร้างโมเดลที่เรียนรู้รูปแบบของสิ่งมีชีวิตในตู้ปลา เพื่อให้นำไปใช้วัดผลและตรวจจับได้จริง
* **คำสั่งสำคัญ:**
  ```python
  model = YOLO('yolov8n.pt')
  model.train(
      data='dataset/data.yaml',
      epochs=25,
      imgsz=640,
      batch=8,
      device='cpu',
      name='aquarium_model',
  )
  ```

---

### 2. สคริปต์ประเมินประสิทธิภาพ (`evaluate.py`)
* **หน้าที่:** ดึงไฟล์ Weights ที่ดีที่สุด (`best.pt`) มาทดสอบกับชุดข้อมูลทดสอบ (`test split`)
* **ความจำเป็น:** ใช้ตรวจสอบความแม่นยำของโมเดลผ่านค่าทางสถิติ เช่น Precision, Recall และ mAP เพื่อประเมินว่าโมเดลทำงานได้ดีเพียงใดก่อนนำไปใช้งานจริง
* **คำสั่งสำคัญ:**
  ```python
  model = YOLO('runs/detect/runs/aquarium_model/weights/best.pt')
  metrics = model.val(data='dataset/data.yaml', split='test', device='cpu')
  ```

---

### 3. สคริปต์ทดสอบกับภาพนิ่ง (`image_test.py`)
* **หน้าที่:** สุ่มดึงภาพจากชุดข้อมูลทดสอบ มาทำการคาดการณ์ (Inference) วาด Bounding Box และแสดงค่าความเชื่อมั่น (Confidence)
* **ความจำเป็น:** ช่วยให้ผู้พัฒนารูปแบบการตรวจจับของโมเดลเห็นภาพจริงแบบ Visual เพื่อตรวจสอบว่าโมเดลระบุวัตถุได้ถูกต้องตรงตามตำแหน่งหรือไม่
* **คำสั่งสำคัญ:**
  ```python
  results = model.predict(source='path/to/image.jpg', conf=0.25, device='cpu')
  annotated_frame = results[0].plot()
  ```

---

### 4. สคริปต์ตรวจจับแบบ Real-time (`realtime_test.py`)
* **หน้าที่:** เปิดกล้อง Webcam และประมวลผลวิดีโอเฟรมต่อเฟรมเพื่อตรวจจับวัตถุแบบทันทีทันใด
* **ความจำเป็น:** ใช้ทดสอบสมรรถนะการทำงานจริงในรูปแบบ Live Stream ว่าโมเดลสามารถประมวลผลได้เร็วพอสำหรับงาน Real-time หรือไม่
* **คำสั่งสำคัญ:**
  ```python
  cap = cv2.VideoCapture(0)
  results = model.predict(source=frame, conf=0.25, device='cpu')
  ```

---

## ขั้นตอนการใช้งาน (How to Run)

1. **เริ่มฝึกสอนโมเดล:**
   ```bash
   py train.py
   ```
2. **ประเมินประสิทธิภาพโมเดล:**
   ```bash
   py evaluate.py
   ```
3. **ทดสอบตรวจจับบนภาพนิ่ง:**
   ```bash
   py image_test.py
   ```
   *(กดปุ่มใดๆ เพื่อดูรูปถัดไป / กด `q` เพื่อออก)*
4. **ทดสอบตรวจจับ Real-time ผ่านกล้อง:**
   ```bash
   py realtime_test.py
   ```
   *(กด `q` เพื่อปิดกล้อง)*

---

## การตั้งค่าโมเดล (Model Configuration)

* **Base Model:** YOLOv8n (3.13M Parameters)
* **Epochs:** 25
* **Image Resolution:** $640 \times 640$ pixels
* **Batch Size:** 8
* **Hardware Execution:** CPU
* **Confidence Threshold:** 0.25

---

## สรุปผลการประเมินและการทำงาน (Evaluation & Results)

ผลการประเมินประสิทธิภาพของโมเดลจากการฝึกสอน 25 Epochs แยกตามคลาส:

| Class | Images | Instances | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All (รวมทุกคลาส)** | **122** | **881** | **0.648** | **0.495** | **0.532** | **0.267** |
| `jellyfish` | 9 | 155 | 0.895 | 0.711 | 0.860 | 0.424 |
| `starfish` | 16 | 26 | 0.711 | 0.692 | 0.705 | 0.407 |
| `stingray` | 23 | 33 | 0.668 | 0.606 | 0.581 | 0.396 |
| `fish` | 61 | 441 | 0.752 | 0.299 | 0.476 | 0.206 |
| `penguin` | 17 | 104 | 0.480 | 0.481 | 0.484 | 0.151 |
| `puffin` | 14 | 71 | 0.559 | 0.303 | 0.324 | 0.145 |
| `shark` | 26 | 51 | 0.470 | 0.373 | 0.294 | 0.140 |

### วิเคราะห์สรุปผล
1. **คลาสที่มีประสิทธิภาพสูง:** `jellyfish` (mAP50 = 0.860) และ `starfish` (mAP50 = 0.705) เนื่องจากมีลักษณะเด่นชัดเจน
2. **คลาสที่ต้องปรับปรุง:** `shark` และ `puffin` เนื่องจากจำนวนตัวอย่างข้อมูลน้อยและมักถูกบดบัง
3. **สรุปภาพรวม:** โมเดลทำค่า mAP50 ได้ **0.532 (53.2%)** ซึ่งถือว่าอยู่ในเกณฑ์ดีสำหรับการใช้โมเดลขนาดเล็ก (Nano) ประมวลผลบน CPU