#!/usr/bin/env python3
"""
infer_vehicle_detector.py  —  Uji model hasil training dengan OpenCV
======================================================================

Script ini memverifikasi bahwa model .onnx atau .pth hasil training dapat
mendeteksi kendaraan secara benar SEBELUM diintegrasikan ke kode C++.

CARA PENGGUNAAN:

  # Uji pada satu gambar (menggunakan model ONNX via OpenCV DNN)
  python3 infer_vehicle_detector.py \\
      --model  ./output/vehicle_detector.onnx \\
      --image  ./dataset/images/foto_001.jpg

  # Uji pada folder gambar (tampilkan satu per satu, tekan tombol apa pun)
  python3 infer_vehicle_detector.py \\
      --model  ./output/vehicle_detector.onnx \\
      --folder ./dataset/images

  # Uji dengan webcam secara real-time
  python3 infer_vehicle_detector.py \\
      --model  ./output/vehicle_detector.onnx \\
      --camera 0

  # Ubah threshold confidence (default 0.40)
  python3 infer_vehicle_detector.py \\
      --model      ./output/vehicle_detector.onnx \\
      --camera     0 \\
      --confidence 0.35

INSTALASI:
  pip install opencv-python-headless numpy
  (atau opencv-python jika ingin GUI)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError as e:
    print(f"\n[ERROR] {e}")
    print("Jalankan: pip install opencv-python numpy\n")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Kelas dan warna bounding box
# ─────────────────────────────────────────────────────────────────────────────

VEHICLE_CLASSES: List[str] = ["background", "car", "motorbike", "bus", "bicycle"]

# BGR colors per kelas
CLASS_COLORS: dict = {
    "car":       (0,   200,  50),   # hijau
    "motorbike": (0,   100, 255),   # oranye
    "bus":       (220,  50,  50),   # biru
    "bicycle":   (200,   0, 200),   # ungu
}


# ─────────────────────────────────────────────────────────────────────────────
# Detektor
# ─────────────────────────────────────────────────────────────────────────────

class VehicleDetectorONNX:
    """Detektor kendaraan menggunakan model ONNX via OpenCV DNN.

    Args:
        model_path:  Path ke file .onnx hasil training.
        img_size:    Ukuran input model (piksel, default 320).
        confidence:  Threshold confidence minimum (default 0.40).
    """

    def __init__(
        self,
        model_path: str,
        img_size:   int   = 320,
        confidence: float = 0.40,
    ) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

        self.img_size   = img_size
        self.confidence = confidence
        self.classes    = VEHICLE_CLASSES

        # Muat model ONNX dengan OpenCV DNN
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        print(f"[Model] Dimuat: {model_path}  (confidence ≥ {confidence})")

    def detect(self, frame: np.ndarray) -> List[dict]:
        """Jalankan deteksi pada satu frame BGR.

        Returns:
            List deteksi, masing-masing berisi:
              'label'      (str),
              'confidence' (float),
              'box'        (x, y, w, h dalam piksel).
        """
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 127.5,
            size=(self.img_size, self.img_size),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)

        # Output tergantung arsitektur — coba ambil semua output layer
        out_names  = self.net.getUnconnectedOutLayersNames()
        outputs    = self.net.forward(out_names)

        results: List[dict] = []

        # Format output SSD: [1, 1, N, 7] — [batch, 1, det_idx, fields]
        # fields: [batch_id, class_id, confidence, x1, y1, x2, y2] (rel.)
        for output in outputs:
            # Normalkan dimensi ke (N, 7) jika perlu
            if output.ndim == 4:
                output = output.reshape(-1, output.shape[-1])

            if output.ndim != 2 or output.shape[1] < 6:
                continue

            for det in output:
                score = float(det[2]) if output.shape[1] >= 7 else float(det[1])
                if score < self.confidence:
                    continue

                class_id = int(det[1]) if output.shape[1] >= 7 else int(det[0])
                if class_id <= 0 or class_id >= len(self.classes):
                    continue

                if output.shape[1] >= 7:
                    x1 = int(det[3] * w)
                    y1 = int(det[4] * h)
                    x2 = int(det[5] * w)
                    y2 = int(det[6] * h)
                else:
                    x1 = int(det[2] * w)
                    y1 = int(det[3] * h)
                    x2 = int(det[4] * w)
                    y2 = int(det[5] * h)

                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                bw = x2 - x1
                bh = y2 - y1
                if bw < 10 or bh < 10:
                    continue

                results.append({
                    "label":      self.classes[class_id],
                    "confidence": score,
                    "box":        (x1, y1, bw, bh),
                })

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualisasi
# ─────────────────────────────────────────────────────────────────────────────

def draw_detections(frame: np.ndarray, detections: List[dict]) -> np.ndarray:
    """Gambar bounding box dan label pada frame."""
    out = frame.copy()
    for det in detections:
        label    = det["label"]
        conf     = det["confidence"]
        x, y, bw, bh = det["box"]
        color    = CLASS_COLORS.get(label, (200, 200, 200))
        text     = f"{label} {conf:.2f}"

        # Bounding box
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)

        # Label background
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 6, y), color, cv2.FILLED)
        cv2.putText(out, text, (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Tampilkan jumlah deteksi
    info = f"Terdeteksi: {len(detections)} objek"
    cv2.putText(out, info, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(out, info, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Mode uji
# ─────────────────────────────────────────────────────────────────────────────

def run_image(detector: VehicleDetectorONNX, image_path: str) -> None:
    """Uji pada satu gambar."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Gagal membuka gambar: {image_path}")
        return

    detections = detector.detect(frame)
    out        = draw_detections(frame, detections)

    print(f"\n[Hasil] {image_path}")
    if detections:
        for d in detections:
            x, y, bw, bh = d["box"]
            print(f"  {d['label']:12s}  conf={d['confidence']:.3f}  "
                  f"box=[{x},{y},{bw},{bh}]")
    else:
        print("  Tidak ada kendaraan terdeteksi.")

    cv2.imshow("Hasil Deteksi — tekan tombol apa pun untuk lanjut", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_folder(detector: VehicleDetectorONNX, folder: str) -> None:
    """Uji pada semua gambar di sebuah folder."""
    exts  = (".jpg", ".jpeg", ".png", ".bmp")
    paths = sorted(
        p for p in Path(folder).iterdir()
        if p.suffix.lower() in exts
    )

    if not paths:
        print(f"[ERROR] Tidak ada gambar di folder: {folder}")
        return

    print(f"\n[Folder] {len(paths)} gambar ditemukan — tekan tombol apa pun / 'q' untuk keluar\n")

    for i, path in enumerate(paths, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        detections = detector.detect(frame)
        out        = draw_detections(frame, detections)

        title = f"[{i}/{len(paths)}] {path.name} — tekan tombol apa pun"
        cv2.imshow(title, out)

        n_det = len(detections)
        det_str = ", ".join(
            f"{d['label']}({d['confidence']:.2f})" for d in detections
        ) or "—"
        print(f"  [{i:3d}/{len(paths)}] {path.name:40s}  {n_det} deteksi: {det_str}")

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord("q"):
            print("  [INFO] Keluar dari loop folder.")
            break


def run_camera(detector: VehicleDetectorONNX, camera_id: int) -> None:
    """Uji secara real-time dari webcam."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka kamera {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\n[Kamera] Membuka kamera {camera_id} — tekan 'q' untuk keluar\n")

    import time
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame kamera.")
            break

        detections = detector.detect(frame)
        out        = draw_detections(frame, detections)

        # FPS
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(out, f"FPS: {fps:.1f}", (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

        cv2.imshow("Vehicle Detector — tekan 'q' untuk keluar", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Uji model vehicle detector ONNX dengan OpenCV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",      required=True, help="Path ke .onnx hasil training")
    p.add_argument("--image",      default=None,  help="Uji pada satu gambar")
    p.add_argument("--folder",     default=None,  help="Uji pada semua gambar di folder")
    p.add_argument("--camera",     type=int, default=None,
                   help="Uji real-time dari webcam (nomor device, misal 0)")
    p.add_argument("--confidence", type=float, default=0.40,
                   help="Threshold confidence minimum")
    p.add_argument("--img-size",   type=int, default=320,
                   help="Ukuran input model (harus sama dengan saat training)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.image is None and args.folder is None and args.camera is None:
        print(
            "[ERROR] Tentukan salah satu dari: --image, --folder, atau --camera\n"
            "Contoh:\n"
            "  python3 infer_vehicle_detector.py --model ./output/vehicle_detector.onnx "
            "--camera 0"
        )
        sys.exit(1)

    detector = VehicleDetectorONNX(
        model_path=args.model,
        img_size=args.img_size,
        confidence=args.confidence,
    )

    if args.image:
        run_image(detector, args.image)
    elif args.folder:
        run_folder(detector, args.folder)
    elif args.camera is not None:
        run_camera(detector, args.camera)


if __name__ == "__main__":
    main()
