#!/usr/bin/env python3
import os
import sys

# Inisialisasi: Pastikan library ultralytics terinstall
try:
    import ultralytics
except ImportError:
    print("[Info] Library 'ultralytics' tidak ditemukan. Menginstall otomatis...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    import ultralytics

from ultralytics import YOLO
import torch

def main():
    print("=" * 60)
    print("  YOLOv8 VEHICLE DETECTOR TRAINING FROM SCRATCH")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device     : {device.upper()}")
    if device == "cuda":
        print(f"  GPU Name   : {torch.cuda.get_device_name(0)}")
    print(f"  Dataset    : yolo_dataset/dataset.yaml")
    print("=" * 60 + "\n")
    
    # 1. Inisialisasi model YOLOv8n (Nano) - sangat cepat untuk edge device/Pi
    # Kita menggunakan transfer learning dari model pretrained yolov8n.pt
    model = YOLO("yolov8n.pt")
    
    # 2. Mulai Pelatihan
    # imgsz=320 sesuai target resolusi Raspberry Pi Anda
    model.train(
        data="yolo_dataset/dataset.yaml",
        epochs=100,                 # Tingkatkan ke 100 epoch untuk konvergensi penuh
        imgsz=320,                  # Resolusi piksel input
        batch=32,                   # Naikkan ke batch 32 untuk stabilitas & kecepatan GPU RTX 2060
        device=0 if device == "cuda" else "cpu",
        workers=4,
        save=True,
        project="yolo_output",      # Folder output hasil training
        name="vehicle_detector"
    )
    
    print("\n[Pelatihan Selesai] Mengambil model terbaik untuk diekspor...")
    
    # 3. Muat model terbaik yang telah dilatih (Ultralytics default menyimpan di runs/detect/)
    possible_paths = [
        os.path.join("runs", "detect", "yolo_output", "vehicle_detector", "weights", "best.pt"),
        os.path.join("yolo_output", "vehicle_detector", "weights", "best.pt"),
        os.path.join("runs", "detect", "train", "weights", "best.pt"),
    ]
    
    best_model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            best_model_path = path
            break
            
    if best_model_path:
        trained_model = YOLO(best_model_path)
        
        # 4. Ekspor ke ONNX dengan Opset 12 (sangat stabil untuk OpenCV DNN C++)
        print(f"\n[Ekspor] Mengekspor {best_model_path} ke ONNX...")
        onnx_path = trained_model.export(format="onnx", imgsz=320, opset=12)
        
        # Salin ke folder output utama agar mudah diakses
        dest_onnx = os.path.join("output", "yolov8n_vehicle.onnx")
        os.makedirs("output", exist_ok=True)
        import shutil
        shutil.copy(onnx_path, dest_onnx)
        
        print("\n" + "="*60)
        print("  PROSES SELESAI DENGAN SUKSES!")
        print(f"  Model YOLOv8 ONNX siap di: {dest_onnx}")
        print("="*60)
    else:
        print(f"[Error] File best.pt tidak ditemukan di path manapun: {possible_paths}")

if __name__ == "__main__":
    main()
