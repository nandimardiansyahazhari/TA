import cv2
import torch
import argparse
from torchvision.transforms import functional as TF
import numpy as np
import os

# Import build_model dari script training Anda
from train_vehicle_detector import build_model, VEHICLE_CLASSES

def parse_args():
    parser = argparse.ArgumentParser(description="Demo Deteksi Kendaraan pada Video")
    parser.add_argument("--video", type=str, required=True, help="/videos_input/video1.mp4")
    parser.add_argument("--model", type=str, default="output/vehicle_detector_best.pth", help="Path ke file model .pth")
    parser.add_argument("--output", type=str, default="output_video.mp4", help="Path untuk menyimpan hasil video")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold (0.0 - 1.0)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.video):
        print(f"[Error] Video tidak ditemukan: {args.video}")
        return
        
    if not os.path.exists(args.model):
        print(f"[Error] Model tidak ditemukan: {args.model}")
        return

    print("[Info] Memuat model PyTorch...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Buat model (jumlah kelas sesuai training)
    model = build_model(num_classes=len(VEHICLE_CLASSES), pretrained_backbone=False)
    
    # 2. Muat bobot dari hasil training terbaik
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    print("[Info] Model berhasil dimuat!")

    # 3. Buka Video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[Error] Gagal membuka video: {args.video}")
        return

    # Siapkan VideoWriter untuk menyimpan hasil
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    # Validasi FPS (Beberapa codec video di Linux mengembalikan nilai FPS aneh seperti 0, 1000, atau nan)
    if fps <= 0 or fps > 120 or np.isnan(fps):
        print(f"[Peringatan] Deteksi FPS video tidak valid ({fps}). Menggunakan fallback 30.0 FPS.")
        fps = 30.0
    else:
        print(f"[Info] Video FPS terdeteksi: {fps}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        
    print(f"\n[Mulai] Memproses video... Tekan 'q' untuk berhenti.")
    print(f"        Hasil akan disimpan ke: {args.output}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
            
        # 4. Pre-process gambar untuk model PyTorch
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = TF.to_tensor(img_rgb).to(device)
        
        # 5. Lakukan deteksi
        with torch.no_grad():
            predictions = model([img_tensor])[0]
            
        # 6. Gambar kotak merah jika yakin > threshold
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        
        for box, score, label_id in zip(boxes, scores, labels):
            if score > args.conf:
                x1, y1, x2, y2 = map(int, box)
                class_name = VEHICLE_CLASSES[label_id]
                
                # Gambar kotak (warna biru untuk mobil, merah untuk motor, dll)
                color = (0, 255, 0) # Hijau default
                if class_name == "car":
                    color = (255, 0, 0) # Biru
                elif class_name == "motor":
                    color = (0, 0, 255) # Merah
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Tulis nama
                text = f"{class_name}: {score*100:.0f}%"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 7. Simpan frame ke video output
        out_video.write(frame)
        
        # 8. Tampilkan ke layar (opsional, bisa di-comment jika ingin lebih cepat)
        cv2.imshow("Demo Deteksi Video", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[Info] Dihentikan oleh pengguna.")
            break
            
        if frame_count % 30 == 0:
            print(f"  Memproses frame {frame_count}...")

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print(f"\n[Selesai] Video hasil deteksi telah disimpan di: {args.output}")

if __name__ == "__main__":
    main()
