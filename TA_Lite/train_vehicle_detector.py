#!/usr/bin/env python3
"""
=============================================================================
TRAIN VEHICLE DETECTOR — Pelatihan Model Deteksi Kendaraan
=============================================================================

Script untuk melatih model deteksi kendaraan (mobil, motor, bus, sepeda)
menggunakan transfer learning dari foto-foto yang sudah diberi anotasi
bounding box.  Model yang dihasilkan kompatibel dengan OpenCV DNN sehingga
dapat langsung digunakan oleh program utama (object_detector.cpp).

─────────────────────────────────────────────────────────────────────────────
FORMAT DATASET  (PASCAL VOC XML — dibuat dengan aplikasi LabelImg)
─────────────────────────────────────────────────────────────────────────────

  dataset/
  ├── images/           ← foto-foto kendaraan  (JPG / PNG)
  │   ├── foto_001.jpg
  │   ├── foto_002.jpg
  │   └── ...
  └── annotations/      ← file anotasi XML, satu file per gambar
      ├── foto_001.xml
      ├── foto_002.xml
      └── ...

  Contoh isi file XML (format LabelImg / PASCAL VOC):

      <annotation>
        <filename>foto_001.jpg</filename>
        <size>
          <width>640</width>
          <height>480</height>
          <depth>3</depth>
        </size>
        <object>
          <name>car</name>
          <bndbox>
            <xmin>120</xmin>  <ymin>80</ymin>
            <xmax>350</xmax>  <ymax>250</ymax>
          </bndbox>
        </object>
        <object>
          <name>motorbike</name>
          <bndbox>
            <xmin>400</xmin>  <ymin>100</ymin>
            <xmax>550</xmax>  <ymax>300</ymax>
          </bndbox>
        </object>
      </annotation>

─────────────────────────────────────────────────────────────────────────────
KELAS YANG DAPAT DIDETEKSI
─────────────────────────────────────────────────────────────────────────────

  Index  Nama kelas   Alias yang diterima
  ─────  ──────────   ────────────────────────────────────────────
    0    background   (otomatis, tidak perlu di anotasi)
    1    car          mobil
    2    motorbike    motor, motorcycle
    3    bus          bis, truk, truck
    4    bicycle      sepeda, bike

─────────────────────────────────────────────────────────────────────────────
INSTALASI DEPENDENSI
─────────────────────────────────────────────────────────────────────────────

  pip install torch torchvision pillow numpy tqdm

─────────────────────────────────────────────────────────────────────────────
CARA PENGGUNAAN
─────────────────────────────────────────────────────────────────────────────

  # Training dasar (50 epoch, batch 4)
  python3 train_vehicle_detector.py --dataset ./dataset

  # Training dengan parameter kustom
  python3 train_vehicle_detector.py  \\
      --dataset  ./dataset           \\
      --output   ./output            \\
      --epochs   100                 \\
      --batch    2                   \\
      --lr       0.005

  # Lanjutkan dari checkpoint
  python3 train_vehicle_detector.py  \\
      --dataset  ./dataset           \\
      --resume   ./output/checkpoint_epoch_30.pth

─────────────────────────────────────────────────────────────────────────────
OUTPUT
─────────────────────────────────────────────────────────────────────────────

  output/vehicle_detector_best.pth   ← checkpoint model terbaik (PyTorch)
  output/vehicle_detector_last.pth   ← checkpoint epoch terakhir
  output/vehicle_detector.onnx       ← model siap pakai di OpenCV DNN
  output/vehicle_classes.txt         ← daftar nama kelas (untuk referensi C++)

─────────────────────────────────────────────────────────────────────────────
PEMBARUAN KODE C++  (object_detector.cpp)  SETELAH TRAINING SELESAI
─────────────────────────────────────────────────────────────────────────────

  1. Constructor — ganti readNetFromCaffe:

       // LAMA:
       net_ = cv::dnn::readNetFromCaffe(configPath, modelPath);
       // BARU:
       net_ = cv::dnn::readNetFromONNX(modelPath);   // hanya 1 argumen

  2. Preprocessing blob — ganti scale dan mean:

       // LAMA (Caffe MobileNet-SSD):
       cv::dnn::blobFromImage(frame, 0.007843, cv::Size(300,300),
                              cv::Scalar(127.5,127.5,127.5), false, false)
       // BARU (model ONNX baru):
       cv::dnn::blobFromImage(frame, 1.0/127.5, cv::Size(320,320),
                              cv::Scalar(127.5,127.5,127.5), true, false)

  3. Class ID — indeks kelas berubah:

       constexpr int kCarClassId       = 1;   // sebelumnya 7
       constexpr int kMotorbikeClassId = 2;   // sebelumnya 14

=============================================================================
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Cek dependensi
# ─────────────────────────────────────────────────────────────────────────────

_missing: list[str] = []
try:
    import torch
    import torch.utils.data
    from torch.utils.data import DataLoader, Subset
    import torchvision
    from torchvision.transforms import functional as TF
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as _e:
    _missing.append(str(_e))

if _missing:
    print("\n[ERROR] Dependensi berikut tidak ditemukan:\n")
    for m in _missing:
        print(f"  {m}")
    print("\nJalankan perintah berikut untuk menginstal:\n")
    print("  pip install torch torchvision pillow numpy tqdm matplotlib\n")
    sys.exit(1)

# tqdm opsional
try:
    from tqdm import tqdm as _tqdm_cls
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI KELAS
# ─────────────────────────────────────────────────────────────────────────────

VEHICLE_CLASSES: List[str] = ["background", "car", "motor"]
CLASS_TO_IDX:    Dict[str, int] = {cls: idx for idx, cls in enumerate(VEHICLE_CLASSES)}
NUM_CLASSES:     int = len(VEHICLE_CLASSES)   # 3

# Alias bahasa Indonesia dan sinonim umum
CLASS_ALIASES: Dict[str, str] = {
    "mobil":       "car",
    "motorcycle":  "motor", 
    "motorbike":   "motor",
    "kendaraan":   "car"
}


def normalize_class(raw: str) -> Optional[str]:
    """Normalisasi nama kelas termasuk alias bahasa Indonesia.

    Returns:
        Nama kelas canonical, atau None jika tidak dikenali.
    """
    name = (raw or "").strip().lower()
    name = CLASS_ALIASES.get(name, name)
    return name if name in CLASS_TO_IDX else None


# ─────────────────────────────────────────────────────────────────────────────
# DATASET — PASCAL VOC XML
# ─────────────────────────────────────────────────────────────────────────────

class VehicleDataset(torch.utils.data.Dataset):
    """Dataset foto kendaraan dengan anotasi PASCAL VOC XML dari LabelImg.

    Args:
        images_dir:      Folder berisi file gambar (JPG / PNG).
        annotations_dir: Folder berisi file anotasi .xml (dari LabelImg).
        transforms:      Optional callable(img_tensor, target) -> (img, target).
    """

    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        transforms=None,
    ) -> None:
        self.images_dir      = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transforms      = transforms
        self.samples: List[Tuple[Path, Path]] = []
        self._scan_files()

    # ── scanning ─────────────────────────────────────────────────────────────

    def _scan_files(self) -> None:
        """Pindai dan pasangkan file gambar dengan file anotasi XML."""
        xml_files = sorted(self.annotations_dir.glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(
                f"Tidak ada file .xml di: {self.annotations_dir}\n"
                "Pastikan folder annotations/ berisi file hasil ekspor LabelImg."
            )

        img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")
        skipped = 0

        for xml_path in xml_files:
            # Cari gambar dengan nama yang sama (berbeda ekstensi)
            img_path: Optional[Path] = None
            for ext in img_extensions:
                candidate = self.images_dir / (xml_path.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                print(f"  [SKIP] Gambar tidak ditemukan untuk: {xml_path.name}")
                skipped += 1
                continue

            # Hanya masukkan jika ada minimal satu objek kendaraan yang valid
            if self._count_valid_objects(xml_path) > 0:
                self.samples.append((img_path, xml_path))
            else:
                print(f"  [SKIP] Tidak ada kelas kendaraan valid di: {xml_path.name}")
                skipped += 1

        total = len(self.samples) + skipped
        print(
            f"\n[Dataset] Ditemukan {len(self.samples)} sampel valid "
            f"dari {total} file XML  ({skipped} dilewati)"
        )

        if len(self.samples) == 0:
            raise ValueError(
                "Dataset kosong!  Periksa hal berikut:\n"
                "  1. Nama file gambar harus sama dengan nama di dalam XML\n"
                f"  2. Nama kelas di XML harus salah satu dari:\n"
                f"       {VEHICLE_CLASSES[1:]}\n"
                f"     atau alias: mobil, motor, bis, sepeda, truck, motorcycle"
            )

    @staticmethod
    def _count_valid_objects(xml_path: Path) -> int:
        """Hitung jumlah objek dengan kelas yang valid di file XML."""
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            return 0
        count = 0
        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is not None and normalize_class(name_el.text or "") is not None:
                count += 1
        return count

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, xml_path = self.samples[idx]

        # Buka gambar dan konversi ke tensor float [0..1] CHW
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img_tensor: torch.Tensor = TF.to_tensor(img)

        # Parse anotasi XML
        boxes:  List[List[float]] = []
        labels: List[int]         = []

        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as e:
            raise RuntimeError(f"Gagal parse XML: {xml_path}") from e

        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is None:
                continue
            class_name = normalize_class(name_el.text or "")
            if class_name is None:
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            try:
                xmin = float(bndbox.findtext("xmin", "0"))
                ymin = float(bndbox.findtext("ymin", "0"))
                xmax = float(bndbox.findtext("xmax", "0"))
                ymax = float(bndbox.findtext("ymax", "0"))
            except ValueError:
                continue

            # Clamp ke batas gambar
            xmin = max(0.0, min(xmin, float(w)))
            ymin = max(0.0, min(ymin, float(h)))
            xmax = max(0.0, min(xmax, float(w)))
            ymax = max(0.0, min(ymax, float(h)))

            # Abaikan bounding box yang tidak valid / terlalu kecil
            if xmax - xmin < 2.0 or ymax - ymin < 2.0:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[class_name])

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t}

        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


def collate_fn(batch):
    """Collate list gambar dengan jumlah objek berbeda-beda."""
    images, targets = zip(*batch)
    return list(images), list(targets)


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTASI DATA
# ─────────────────────────────────────────────────────────────────────────────

class RandomHorizontalFlip:
    """Flip horizontal acak dengan penyesuaian koordinat bounding box."""

    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, img: torch.Tensor, target: dict):
        if random.random() < self.prob:
            img = TF.hflip(img)
            width = img.shape[-1]
            if target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return img, target


class ColorJitter:
    """Perubahan kecerahan, kontras, saturasi, dan hue secara acak."""

    def __init__(
        self,
        brightness: float = 0.3,
        contrast:   float = 0.3,
        saturation: float = 0.3,
        hue:        float = 0.05,
    ) -> None:
        self._jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue,
        )

    def __call__(self, img: torch.Tensor, target: dict):
        img_pil = TF.to_pil_image(img)
        img_pil = self._jitter(img_pil)
        return TF.to_tensor(img_pil), target


class Compose:
    """Gabungkan beberapa transform yang menerima (img, target)."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


def get_train_transforms() -> Compose:
    return Compose([
        RandomHorizontalFlip(prob=0.5),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — SSDLite320 MobileNetV3-Large  (Transfer Learning)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, pretrained_backbone: bool = True) -> torch.nn.Module:
    """Buat SSDLite320-MobileNetV3 dengan transfer learning backbone ImageNet.

    Args:
        num_classes:         Jumlah kelas termasuk background (misal 5).
        pretrained_backbone: Jika True, backbone di-init dari bobot ImageNet.

    Returns:
        torch.nn.Module siap dilatih.
    """
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large

    print(
        f"[Model] Membangun SSDLite320-MobileNetV3  "
        f"({num_classes} kelas, pretrained_backbone={pretrained_backbone})"
    )

    # ── Buat model kosong dengan jumlah kelas kustom ──────────────────────
    try:
        # torchvision >= 0.13
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )
    except TypeError:
        # torchvision < 0.13  (API lama)
        model = ssdlite320_mobilenet_v3_large(
            pretrained=False,
            pretrained_backbone=False,
            num_classes=num_classes,
        )

    # ── Transfer bobot backbone dari model pretrained ImageNet ────────────
    if pretrained_backbone:
        try:
            try:
                pretrained = ssdlite320_mobilenet_v3_large(
                    weights="DEFAULT", weights_backbone="DEFAULT"
                )
            except TypeError:
                pretrained = ssdlite320_mobilenet_v3_large(pretrained=True)

            backbone_state = {
                k: v for k, v in pretrained.state_dict().items()
                if k.startswith("backbone.")
            }
            missing, unexpected = model.load_state_dict(backbone_state, strict=False)
            print(
                f"  Backbone transferred — "
                f"missing: {len(missing)}, unexpected: {len(unexpected)}"
            )
            del pretrained
        except Exception as e:
            print(f"  [PERINGATAN] Gagal transfer backbone: {e}")
            print("  Training akan dilanjutkan tanpa pretrained backbone.")

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameter total: {total_params:,}  |  yang dilatih: {train_params:,}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & VALIDASI
# ─────────────────────────────────────────────────────────────────────────────

def _iter_loader(loader, desc: str):
    """Iterasi dataloader dengan tqdm (jika tersedia) atau tanpa."""
    if HAS_TQDM:
        return _tqdm_cls(loader, desc=desc, unit="batch", leave=False)
    return loader


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> float:
    """Satu epoch training; mengembalikan rata-rata total loss."""
    model.train()
    total_loss  = 0.0
    num_batches = 0

    for images, targets in _iter_loader(dataloader, f"Epoch {epoch+1}/{num_epochs} [Train]"):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Lewati batch yang semuanya tidak punya objek terdeteksi
        if all(t["boxes"].numel() == 0 for t in targets):
            continue

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()

        # Gradient clipping untuk stabilitas
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()
        total_loss  += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Hitung rata-rata validation loss."""
    model.train()   # mode train agar SSD mengembalikan loss, bukan prediksi
    total_loss  = 0.0
    num_batches = 0

    for images, targets in _iter_loader(dataloader, "Validasi"):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if all(t["boxes"].numel() == 0 for t in targets):
            continue

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        total_loss  += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


# ─────────────────────────────────────────────────────────────────────────────
# EKSPOR MODEL
# ─────────────────────────────────────────────────────────────────────────────

class _SSDWrapper(torch.nn.Module):
    """Wrapper agar SSD menerima satu tensor (bukan list) — diperlukan ONNX."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        return self.model(list(x))


def export_onnx(
    model: torch.nn.Module,
    output_path: str,
    img_size: int = 320,
) -> bool:
    """Ekspor model ke ONNX untuk digunakan di OpenCV DNN.

    Returns:
        True jika berhasil, False jika gagal.
    """
    print(f"\n[Ekspor] Menyimpan model ONNX → {output_path}")
    model.eval()
    model_cpu = model.to("cpu")

    wrapper = _SSDWrapper(model_cpu)
    wrapper.eval()

    dummy = torch.zeros(1, 3, img_size, img_size)

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy,
                output_path,
                opset_version=11,
                input_names=["input"],
                export_params=True,
                verbose=False,
                do_constant_folding=True,
            )
        print(f"  [OK] ONNX berhasil disimpan: {output_path}")
        return True
    except Exception as e:
        print(f"  [GAGAL] Export ONNX gagal: {e}")
        print("  Model PyTorch tetap tersimpan sebagai .pth untuk inferensi Python.")
        return False


def export_torchscript(
    model: torch.nn.Module,
    output_path: str,
    img_size: int = 320,
) -> bool:
    """Ekspor model ke TorchScript (alternatif jika ONNX gagal)."""
    print(f"[Ekspor] Menyimpan TorchScript → {output_path}")
    model.eval()
    model_cpu = model.to("cpu")
    try:
        dummy = [torch.zeros(3, img_size, img_size)]
        with torch.no_grad():
            scripted = torch.jit.trace(model_cpu, [dummy])
        scripted.save(output_path)
        print(f"  [OK] TorchScript berhasil disimpan: {output_path}")
        return True
    except Exception as e:
        print(f"  [GAGAL] TorchScript gagal: {e}")
        return False


def save_class_labels(output_dir: str) -> None:
    """Simpan daftar nama kelas ke file teks (untuk referensi pembacaan di C++)."""
    path = os.path.join(output_dir, "vehicle_classes.txt")
    with open(path, "w", encoding="utf-8") as f:
        for cls in VEHICLE_CLASSES:
            f.write(cls + "\n")
    print(f"  [OK] Daftar kelas disimpan: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMEN CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Training model deteksi kendaraan dari foto ber-bounding-box",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", default="./dataset",
        help="Folder dataset (harus berisi subfolder images/ dan annotations/)",
    )
    p.add_argument(
        "--output", default="./output",
        help="Folder untuk menyimpan model dan checkpoint",
    )
    p.add_argument(
        "--epochs", type=int, default=50,
        help="Jumlah epoch training",
    )
    p.add_argument(
        "--batch", type=int, default=4,
        help="Ukuran batch  (kurangi menjadi 2 jika RAM tidak mencukupi)",
    )
    p.add_argument(
        "--lr", type=float, default=0.005,
        help="Learning rate awal",
    )
    p.add_argument(
        "--img-size", type=int, default=320,
        help="Ukuran sisi gambar input model (piksel)",
    )
    p.add_argument(
        "--val-split", type=float, default=0.15,
        help="Proporsi data validasi (0.15 = 15%%)",
    )
    p.add_argument(
        "--workers", type=int, default=2,
        help="Jumlah worker DataLoader",
    )
    p.add_argument(
        "--resume", default=None,
        help="Path checkpoint .pth untuk melanjutkan training",
    )
    p.add_argument(
        "--no-pretrained", action="store_true",
        help="Latih dari nol tanpa bobot ImageNet (tidak dianjurkan)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed untuk reproduktibilitas",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Header
    print(f"\n{'='*62}")
    print("  TRAINING VEHICLE DETECTOR")
    print(f"{'='*62}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Output     : {args.output}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch}")
    print(f"  LR         : {args.lr}")
    print(f"  Img size   : {args.img_size}×{args.img_size}")
    print(f"  Device     : {device}")
    print(f"  Kelas      : {VEHICLE_CLASSES}")
    if not HAS_TQDM:
        print("  [INFO] tqdm tidak terinstall — progress bar dinonaktifkan")
    print(f"{'='*62}\n")

    os.makedirs(args.output, exist_ok=True)

    # ── Validasi folder dataset ───────────────────────────────────────────────
    images_dir      = os.path.join(args.dataset, "images")
    annotations_dir = os.path.join(args.dataset, "annotations")

    for folder, label in [(images_dir, "images/"), (annotations_dir, "annotations/")]:
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Folder '{label}' tidak ditemukan di: {args.dataset}\n"
                "Struktur yang diharapkan:\n"
                f"  {args.dataset}/\n"
                "  ├── images/\n"
                "  └── annotations/"
            )

    # ── Dataset — dua instance terpisah agar augmentasi tidak masuk ke val ──
    print("[Dataset] Memuat data training (dengan augmentasi)...")
    train_full = VehicleDataset(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        transforms=get_train_transforms(),
    )

    print("[Dataset] Memuat data validasi (tanpa augmentasi)...")
    val_full = VehicleDataset(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        transforms=None,
    )

    n_total = len(train_full)
    n_val   = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    # Buat indeks split yang konsisten menggunakan seed
    generator = torch.Generator().manual_seed(args.seed)
    indices   = torch.randperm(n_total, generator=generator).tolist()

    train_dataset = Subset(train_full, indices[:n_train])
    val_dataset   = Subset(val_full,   indices[n_train:])

    print(f"\n[Dataset] Train: {n_train} sampel  |  Validasi: {n_val} sampel\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(val_dataset) % args.batch == 1),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes=NUM_CLASSES,
        pretrained_backbone=not args.no_pretrained,
    )
    model.to(device)

    # ── Optimizer — SGD dengan momentum ──────────────────────────────────────
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler: turunkan 10× di epoch ke-60% dan ke-80% training
    milestones = [int(args.epochs * 0.6), int(args.epochs * 0.8)]
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )

    start_epoch   = 0
    best_val_loss = float("inf")

    # ── Resume dari checkpoint ────────────────────────────────────────────────
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Checkpoint tidak ditemukan: {args.resume}")
        print(f"[Resume] Memuat checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Melanjutkan dari epoch {start_epoch}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"[Training] Mulai training {args.epochs} epoch...\n")
    
    history_train_loss = []
    history_val_loss = []

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, args.epochs
        )
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        is_best    = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss

        print(
            f"Epoch [{epoch+1:3d}/{args.epochs}]  "
            f"Train: {train_loss:.4f}  "
            f"Val: {val_loss:.4f}  "
            f"LR: {current_lr:.2e}"
            + ("  ★ best" if is_best else "")
        )

        # Utilitas simpan checkpoint
        def _save_ckpt(path: str) -> None:
            torch.save({
                "epoch":         epoch,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "classes":       VEHICLE_CLASSES,
                "num_classes":   NUM_CLASSES,
                "img_size":      args.img_size,
            }, path)

        # Simpan checkpoint terbaik
        if is_best:
            _save_ckpt(os.path.join(args.output, "vehicle_detector_best.pth"))

        # Simpan checkpoint terakhir setiap epoch
        _save_ckpt(os.path.join(args.output, "vehicle_detector_last.pth"))

        # Checkpoint berkala setiap 10 epoch
        if (epoch + 1) % 10 == 0:
            _save_ckpt(
                os.path.join(args.output, f"checkpoint_epoch_{epoch+1}.pth")
            )
            
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)

    # Buat dan simpan grafik di akhir training
    plt.figure(figsize=(8, 5))
    plt.plot(range(start_epoch + 1, args.epochs + 1), history_train_loss, label="Train Loss", marker='o')
    plt.plot(range(start_epoch + 1, args.epochs + 1), history_val_loss, label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Grafik Loss Training")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output, "training_graph.png"))
    plt.close()

    # ── Ekspor model akhir ────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  TRAINING SELESAI")
    print(f"{'='*62}")
    print(f"  Best val loss : {best_val_loss:.4f}")

    # Muat model terbaik untuk ekspor
    best_ckpt_path = os.path.join(args.output, "vehicle_detector_best.pth")
    if os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"  Menggunakan bobot terbaik dari: {best_ckpt_path}")

    # Ekspor ONNX
    onnx_path = os.path.join(args.output, "vehicle_detector.onnx")
    onnx_ok   = export_onnx(model, onnx_path, img_size=args.img_size)

    # Fallback TorchScript jika ONNX gagal
    if not onnx_ok:
        ts_path = os.path.join(args.output, "vehicle_detector.pt")
        export_torchscript(model, ts_path, img_size=args.img_size)

    # Simpan daftar kelas
    save_class_labels(args.output)

    # ── Ringkasan ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("[Ringkasan file output]")
    for fname in [
        "vehicle_detector_best.pth",
        "vehicle_detector_last.pth",
        "vehicle_detector.onnx",
        "vehicle_classes.txt",
    ]:
        fpath = os.path.join(args.output, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  ✓  {fname:<35} ({size_kb:,.0f} KB)")

    print(f"\n[Instruksi C++]  Edit TA_Lite/object_detector.cpp:")
    print(f"  1. readNetFromCaffe(...)  →  readNetFromONNX(\"vehicle_detector.onnx\")")
    print(f"  2. blobFromImage scale    →  1.0/127.5,  size 320×320")
    print(f"  3. kCarClassId = 1,  kMotorbikeClassId = 2")
    print()


if __name__ == "__main__":
    main()
