#!/usr/bin/env python3
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

# Reproducibility
random.seed(42)

# Konfigurasi Kelas
VEHICLE_CLASSES = ["car", "motor"]  # YOLO tidak menggunakan background class (0: car, 1: motor)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VEHICLE_CLASSES)}

CLASS_ALIASES: Dict[str, str] = {
    "mobil":       "car",
    "motorcycle":  "motor",
    "motorbike":   "motor",
    "kendaraan":   "car",
    "motor":       "motor",
    "truck":       "car",
    "truk":        "car",
    "ambulance":   "car",
    "police car":  "car",
    "police_car":  "car",
}

def normalize_class(raw: str) -> Optional[str]:
    name = (raw or "").strip().lower()
    name = CLASS_ALIASES.get(name, name)
    return name if name in CLASS_TO_IDX else None

def convert_voc_to_yolo(xml_path: Path, img_w: int, img_h: int) -> List[str]:
    yolo_lines = []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
        
    for obj in root.findall("object"):
        cls = normalize_class(obj.findtext("name") or "")
        if cls is None:
            continue
            
        bb = obj.find("bndbox")
        if bb is None:
            continue
            
        xmin = max(0, min(float(bb.findtext("xmin", "0")), img_w))
        ymin = max(0, min(float(bb.findtext("ymin", "0")), img_h))
        xmax = max(0, min(float(bb.findtext("xmax", "0")), img_w))
        ymax = max(0, min(float(bb.findtext("ymax", "0")), img_h))
        
        if xmax - xmin < 2 or ymax - ymin < 2:
            continue
            
        # Perhitungan koordinat YOLO
        dw = 1.0 / img_w
        dh = 1.0 / img_h
        
        cx = (xmin + xmax) / 2.0 * dw
        cy = (ymin + ymax) / 2.0 * dh
        bw = (xmax - xmin) * dw
        bh = (ymax - ymin) * dh
        
        class_id = CLASS_TO_IDX[cls]
        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        
    return yolo_lines

def main():
    src_dir = Path("/home/ansyah/TA-main/TA_Lite/dataset/dataset")
    images_dir = src_dir / "images"
    annotations_dir = src_dir / "annotations"
    
    dest_dir = Path("/home/ansyah/TA-main/TA_Lite/yolo_dataset")
    
    print("[Info] Memindai file dataset...")
    xml_files = sorted(annotations_dir.glob("*.xml"))
    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")
    
    samples = []
    for xml_path in xml_files:
        img_path = None
        for ext in img_exts:
            c = images_dir / (xml_path.stem + ext)
            if c.exists():
                img_path = c
                break
        if img_path:
            samples.append((img_path, xml_path))
            
    print(f"[Dataset] Menemukan {len(samples)} sampel gambar & anotasi.")
    
    # Shuffle dan split
    random.shuffle(samples)
    val_split = 0.15
    n_val = int(len(samples) * val_split)
    
    splits = {
        "train": samples[n_val:],
        "val": samples[:n_val]
    }
    
    # Buat direktori tujuan
    for split in ["train", "val"]:
        os.makedirs(dest_dir / "images" / split, exist_ok=True)
        os.makedirs(dest_dir / "labels" / split, exist_ok=True)
        
    print("\n[Mulai] Mengonversi ke format YOLO...")
    
    counts = {"train": 0, "val": 0}
    for split, sample_list in splits.items():
        for img_path, xml_path in sample_list:
            # Baca ukuran gambar asli menggunakan PIL untuk normalisasi yang akurat
            from PIL import Image
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception:
                continue
                
            # Konversi VOC XML ke baris teks YOLO
            yolo_lines = convert_voc_to_yolo(xml_path, w, h)
            if not yolo_lines:
                continue  # Lewati jika tidak ada objek yang valid
                
            # Salin gambar ke tujuan
            shutil.copy(img_path, dest_dir / "images" / split / img_path.name)
            
            # Tulis anotasi YOLO ke tujuan
            txt_name = xml_path.stem + ".txt"
            with open(dest_dir / "labels" / split / txt_name, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")
                
            counts[split] += 1
            
    print(f"\n[OK] Konversi Selesai!")
    print(f"     Train : {counts['train']} sampel")
    print(f"     Val   : {counts['val']} sampel")
    print(f"     Folder: {dest_dir}")
    
    # Buat file konfigurasi dataset.yaml untuk YOLOv8
    yaml_content = f"""# YOLOv8 Dataset Configuration
path: {dest_dir.absolute()}
train: images/train
val: images/val

names:
  0: car
  1: motor
"""
    with open(dest_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)
        
    print(f"[OK] Menulis file konfigurasi: {dest_dir / 'dataset.yaml'}")

if __name__ == "__main__":
    main()
