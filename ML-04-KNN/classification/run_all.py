import subprocess

files = ["data_loader.py", "evaluate.py", "knn_tf.py","main.py"]

for f in files:
    print(f"กำลังรัน: {f}")
    subprocess.run(["python", f])