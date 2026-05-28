import cv2
import torch
from torchvision.transforms import functional as TF
import numpy as np

# Import build_model dari script training Anda
from train_vehicle_detector import build_model, VEHICLE_CLASSES

def main():
    print("[Info] Memuat model PyTorch...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Buat model (jumlah kelas sesuai training)
    model = build_model(num_classes=len(VEHICLE_CLASSES), pretrained_backbone=False)
    
    # 2. Muat bobot dari hasil training terbaik
    checkpoint = torch.load("output/vehicle_detector_best.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    print("[Info] Model berhasil dimuat!")

    # 3. Buka Kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1) # Coba kamera eksternal jika kamera internal gagal
        
    print("\n[Mulai] Tekan 'q' untuk keluar dari kamera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 4. Pre-process gambar untuk model PyTorch
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = TF.to_tensor(img_rgb).to(device)
        
        # 5. Lakukan deteksi
        with torch.no_grad():
            # Model menerima list tensor gambar
            predictions = model([img_tensor])[0]
            
        # 6. Gambar kotak merah jika yakin > 30%
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        
        for box, score, label_id in zip(boxes, scores, labels):
            if score > 0.3: # Threshold
                x1, y1, x2, y2 = map(int, box)
                class_name = VEHICLE_CLASSES[label_id]
                
                # Gambar kotak
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Tulis nama
                text = f"{class_name}: {score*100:.0f}%"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 7. Tampilkan ke layar
        cv2.imshow("Hasil Deteksi PyTorch NATIVE", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
