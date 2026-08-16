


"""
Read CSV
convert text to number
make Scaling for KNN
split data: train / validation / test
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).resolve().parent.parent / "data-earphone" / "earphone.csv"

TARGET = "Diet_Type"

# config feature is numeric features that are already numbers
NUMERIC_FEATURES = [
    "ratings",
    "reviews",
    "selling_price",
    "actual_price",
    "off"
]

# ---------------------------------------------------------------------------
def load_data(test_size=0.2, seed=42):

    # step 1 : read CSV
    df = pd.read_csv(CSV_PATH)

    # ดรอปแถวที่ไม่มีข้อมูลในคอลัมน์ที่จำเป็นเบื้องต้น
    df = df.dropna(subset=['colours', 'ratings', 'reviews', 'selling_price', 'actual_price', 'off'])
    
    # step 2 : Data Cleaning & Feature Engineering
    # ลบวงเล็บและลูกน้ำออกจากรีวิว แล้วแปลงเป็น float (เช่น "(9,400)" -> 9400.0)
    df['reviews'] = df['reviews'].astype(str).str.replace(r'[(),]', '', regex=True).astype(float)
    
    # ลบสัญลักษณ์ ₹ และลูกน้ำ ออกจากราคา (เช่น "₹1,999" -> 1999.0)
    df['selling_price'] = df['selling_price'].astype(str).str.replace(r'[₹,]', '', regex=True).astype(float)
    df['actual_price'] = df['actual_price'].astype(str).str.replace(r'[₹,]', '', regex=True).astype(float)
    
    # ลบคำว่า "% off" ออกจากคอลัมน์ลดราคา (เช่น "50% off" -> 50.0)
    df['off'] = df['off'].astype(str).str.replace(r'% off', '', regex=True).astype(float)
    
    # สกัด Form Factor ออกมาจากคอลัมน์ colours (เช่น "Black, In the Ear" -> "In the Ear")
    df[TARGET] = df['colours'].astype(str).str.split(', ').str[-1]
    
    # เตรียมข้อมูล Features (X)
    X = df[NUMERIC_FEATURES].copy()

    # convert result (target) to number : สกัดชื่อคลาสและแปลงเป็นตัวเลข
    class_names = sorted(df[TARGET].unique())
    y = df[TARGET].map({name: i for i, name in enumerate(class_names)})

    X = X.to_numpy(dtype="float32")
    y = y.to_numpy(dtype="int32")

    # step 3 : split data เป็น train 60 / validation 20 / test 20
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp)

    # step 4 : Scaling 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "class_names": class_names,
        "feature_names": NUMERIC_FEATURES,
        "n_rows": len(df),
    }

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    print("จำนวนแถวทั้งหมดหลังคลีน :", data["n_rows"])
    print("train :", data["X_train"].shape)
    print("val   :", data["X_val"].shape)
    print("test  :", data["X_test"].shape)
    print("คลาส  :", data["class_names"])