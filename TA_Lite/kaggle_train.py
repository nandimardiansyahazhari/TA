#!/usr/bin/env python3
"""
=============================================================================
KAGGLE TRAINING SCRIPT — Vehicle Detector
=============================================================================
Cara Menggunakan di Kaggle:
1. Upload script ini ke Kaggle Notebook
2. Upload dataset Anda (format: ZIP berisi folder images/ dan annotations/)
3. Aktifkan GPU: Settings > Accelerator > GPU T4 x2
4. Jalankan semua cell satu per satu

Output:
  /kaggle/working/output/vehicle_detector_best.pth
  /kaggle/working/output/training_graph.png
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: Install dependensi (jalankan pertama kali saja)
# ─────────────────────────────────────────────────────────────────────────────
# !pip install -q tqdm matplotlib

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2: Import library
# ─────────────────────────────────────────────────────────────────────────────
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile

import torch
import torch.utils.data
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.transforms import functional as TF
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Tidak ada'}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3: KONFIGURASI — Sesuaikan bagian ini
# ─────────────────────────────────────────────────────────────────────────────

# Path ke dataset Anda di Kaggle (sesuaikan jika perlu)
# Jika upload dataset sebagai ZIP, ekstrak dulu di Cell berikutnya
DATASET_DIR  = "/kaggle/input/your-dataset-name/dataset"  # <-- GANTI INI
OUTPUT_DIR   = "/kaggle/working/output"

# Parameter Training
EPOCHS       = 50
BATCH_SIZE   = 8    # Kaggle GPU bisa handle lebih besar dari laptop
LEARNING_RATE = 0.005
IMG_SIZE     = 320
VAL_SPLIT    = 0.15
SEED         = 42

# Kelas yang dideteksi (JANGAN ubah background di index 0)
VEHICLE_CLASSES: List[str] = ["background", "car", "motor"]
CLASS_TO_IDX   : Dict[str, int] = {cls: idx for idx, cls in enumerate(VEHICLE_CLASSES)}
NUM_CLASSES    : int = len(VEHICLE_CLASSES)

# Alias nama kelas dari dataset internet
CLASS_ALIASES: Dict[str, str] = {
    "mobil":       "car",
    "motorcycle":  "motor",
    "motorbike":   "motor",
    "kendaraan":   "car",
    "motor":       "motor",
}

print(f"Dataset  : {DATASET_DIR}")
print(f"Output   : {OUTPUT_DIR}")
print(f"Kelas    : {VEHICLE_CLASSES}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4: (Opsional) Ekstrak ZIP dataset jika perlu
# ─────────────────────────────────────────────────────────────────────────────
# Hapus tanda # di bawah ini jika dataset Anda berupa file ZIP
#
# ZIP_PATH = "/kaggle/input/your-dataset-name/dataset.zip"
# EXTRACT_TO = "/kaggle/working/dataset"
# with zipfile.ZipFile(ZIP_PATH, 'r') as z:
#     z.extractall(EXTRACT_TO)
# DATASET_DIR = EXTRACT_TO
# print("Ekstraksi selesai!")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5: Dataset & Augmentasi
# ─────────────────────────────────────────────────────────────────────────────

def normalize_class(raw: str) -> Optional[str]:
    name = (raw or "").strip().lower()
    name = CLASS_ALIASES.get(name, name)
    return name if name in CLASS_TO_IDX else None


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, img, target):
        if random.random() < self.prob:
            img = TF.hflip(img)
            w = img.shape[-1]
            if target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return img, target


class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self._jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue,
        )

    def __call__(self, img, target):
        img_pil = TF.to_pil_image(img)
        img_pil = self._jitter(img_pil)
        return TF.to_tensor(img_pil), target


class RandomZoom:
    """Zoom in secara acak (crop & resize)."""
    def __init__(self, scale_range=(0.7, 1.0)):
        self.scale_range = scale_range

    def __call__(self, img, target):
        scale = random.uniform(*self.scale_range)
        _, h, w = img.shape
        new_h, new_w = int(h * scale), int(w * scale)
        top  = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        img = TF.resized_crop(img, top, left, new_h, new_w, (h, w))
        if target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left).clamp(0, w) * (w / new_w)
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top).clamp(0, h) * (h / new_h)
            target["boxes"] = boxes
        return img, target


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


def get_train_transforms():
    return Compose([
        RandomHorizontalFlip(prob=0.5),
        RandomZoom(scale_range=(0.7, 1.0)),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ])


class VehicleDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None):
        self.images_dir      = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transforms      = transforms
        self.samples         = []
        self._scan_files()

    def _scan_files(self):
        xml_files   = sorted(self.annotations_dir.glob("*.xml"))
        img_exts    = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")
        skipped     = 0
        for xml_path in xml_files:
            img_path = None
            for ext in img_exts:
                c = self.images_dir / (xml_path.stem + ext)
                if c.exists():
                    img_path = c
                    break
            if img_path is None:
                skipped += 1
                continue
            if self._count_valid(xml_path) > 0:
                self.samples.append((img_path, xml_path))
            else:
                skipped += 1
        print(f"[Dataset] {len(self.samples)} valid, {skipped} dilewati")
        if not self.samples:
            raise ValueError("Dataset kosong! Periksa nama kelas di XML Anda.")

    @staticmethod
    def _count_valid(xml_path):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            return 0
        return sum(
            1 for obj in root.findall("object")
            if normalize_class((obj.findtext("name") or "")) is not None
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, xml_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img_t = TF.to_tensor(img)

        boxes, labels = [], []
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            cls = normalize_class(obj.findtext("name") or "")
            if cls is None:
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = max(0, min(float(bb.findtext("xmin", "0")), w))
            ymin = max(0, min(float(bb.findtext("ymin", "0")), h))
            xmax = max(0, min(float(bb.findtext("xmax", "0")), w))
            ymax = max(0, min(float(bb.findtext("ymax", "0")), h))
            if xmax - xmin < 2 or ymax - ymin < 2:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[cls])

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t}
        if self.transforms:
            img_t, target = self.transforms(img_t, target)
        return img_t, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6: Fungsi Training & Evaluasi
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, loader, device, epoch, total_epochs):
    model.train()
    total_loss, n_batch = 0.0, 0
    it = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]") if HAS_TQDM else loader
    for images, targets in it:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if all(t["boxes"].numel() == 0 for t in targets):
            continue
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        total_loss += loss.item()
        n_batch    += 1
    return total_loss / max(1, n_batch)


@torch.no_grad()
def evaluate(model, loader, device):
    model.train()
    total_loss, n_batch = 0.0, 0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if all(t["boxes"].numel() == 0 for t in targets):
            continue
        loss_dict = model(images, targets)
        total_loss += sum(loss_dict.values()).item()
        n_batch    += 1
    return total_loss / max(1, n_batch)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7: Mulai Training
# ─────────────────────────────────────────────────────────────────────────────

def run_training():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images_dir      = os.path.join(DATASET_DIR, "images")
    annotations_dir = os.path.join(DATASET_DIR, "annotations")

    print("\n[Dataset] Memuat data training...")
    train_full = VehicleDataset(images_dir, annotations_dir, transforms=get_train_transforms())
    print("[Dataset] Memuat data validasi...")
    val_full   = VehicleDataset(images_dir, annotations_dir, transforms=None)

    n_total = len(train_full)
    n_val   = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val
    gen     = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(n_total, generator=gen).tolist()

    train_loader = DataLoader(
        Subset(train_full, indices[:n_train]),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_full, indices[n_train:]),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
        collate_fn=collate_fn, pin_memory=True,
    )
    print(f"\nTrain: {n_train}  |  Val: {n_val}  |  Device: {device}\n")

    # Model
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=NUM_CLASSES)
    # Transfer backbone dari ImageNet
    try:
        pretrained = ssdlite320_mobilenet_v3_large(weights="DEFAULT", weights_backbone="DEFAULT")
        backbone_state = {k: v for k, v in pretrained.state_dict().items() if k.startswith("backbone.")}
        model.load_state_dict(backbone_state, strict=False)
        del pretrained
        print("[Model] Backbone ImageNet berhasil di-load!")
    except Exception as e:
        print(f"[Model] Backbone tidak di-load: {e}")
    model.to(device)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(EPOCHS*0.6), int(EPOCHS*0.8)], gamma=0.1
    )

    best_val_loss       = float("inf")
    history_train_loss  = []
    history_val_loss    = []

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, EPOCHS)
        val_loss   = evaluate(model, val_loader, device)
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch+1:3d}/{EPOCHS}]  "
            f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  LR: {lr_now:.2e}"
            + ("  ★ BEST" if is_best else "")
        )

        ckpt = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss, "classes": VEHICLE_CLASSES,
        }
        torch.save(ckpt, os.path.join(OUTPUT_DIR, "vehicle_detector_last.pth"))
        if is_best:
            torch.save(ckpt, os.path.join(OUTPUT_DIR, "vehicle_detector_best.pth"))
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))

    # Simpan grafik
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, EPOCHS+1), history_train_loss, label="Train Loss", marker='o', markersize=3)
    plt.plot(range(1, EPOCHS+1), history_val_loss,   label="Val Loss",   marker='o', markersize=3)
    plt.axvline(history_val_loss.index(min(history_val_loss)) + 1,
                color='red', linestyle='--', label=f"Best Epoch")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Grafik Loss Training — Vehicle Detector")
    plt.legend(); plt.grid(True)
    graph_path = os.path.join(OUTPUT_DIR, "training_graph.png")
    plt.savefig(graph_path, dpi=150)
    plt.show()
    print(f"\n[Grafik] Disimpan: {graph_path}")

    # Simpan daftar kelas
    with open(os.path.join(OUTPUT_DIR, "vehicle_classes.txt"), "w") as f:
        f.write("\n".join(VEHICLE_CLASSES) + "\n")

    print(f"\n{'='*50}")
    print(f"  TRAINING SELESAI")
    print(f"  Best Val Loss  : {best_val_loss:.4f}")
    print(f"  Best pada epoch: {history_val_loss.index(min(history_val_loss)) + 1}")
    print(f"  Model tersimpan: {OUTPUT_DIR}/vehicle_detector_best.pth")
    print(f"{'='*50}")
    print("\n>> Download file .pth dari panel Output Kaggle (kanan bawah)")


# Jalankan training
run_training()
