#!/usr/bin/env python3
"""
Cloud-ready vehicle detector training script for Google Colab and Kaggle.

Tujuan:
- Bisa dipakai di Google Colab atau Kaggle dengan GPU gratis.
- Tetap memakai dataset LabelImg / Pascal VOC XML yang sama seperti script lokal.
- Bisa membaca dataset dari folder biasa atau dari file ZIP.
- Menyimpan output model .pth / .onnx ke folder kerja cloud.

Format dataset yang didukung:
  dataset/
  ├── images/
  │   ├── img_001.jpg
  │   └── ...
  └── annotations/
      ├── img_001.xml
      └── ...

Contoh penggunaan di Google Colab:
  !python TA_Lite/train_vehicle_detector_cloud.py \
      --dataset-zip /content/drive/MyDrive/dataset_kendaraan.zip \
      --output /content/drive/MyDrive/vehicle_training_output \
      --epochs 40 \
      --batch 8

Contoh penggunaan di Kaggle:
  !python /kaggle/working/TA/TA_Lite/train_vehicle_detector_cloud.py \
      --dataset-dir /kaggle/input/dataset-kendaraan/dataset \
      --output /kaggle/working/output \
      --epochs 40 \
      --batch 8

Catatan:
- Gunakan Runtime GPU di Colab.
- Gunakan accelerator GPU di Kaggle Notebook.
- Jika RAM/GPU kecil, turunkan batch menjadi 2 atau 4.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path


def _append_script_dir() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


_append_script_dir()

from train_vehicle_detector import export_onnx  # noqa: E402
from train_vehicle_detector import get_train_transforms  # noqa: E402
from train_vehicle_detector import build_model  # noqa: E402
from train_vehicle_detector import collate_fn  # noqa: E402
from train_vehicle_detector import evaluate  # noqa: E402
from train_vehicle_detector import save_class_labels  # noqa: E402
from train_vehicle_detector import train_one_epoch  # noqa: E402
from train_vehicle_detector import VehicleDataset  # noqa: E402
from train_vehicle_detector import NUM_CLASSES  # noqa: E402
from train_vehicle_detector import VEHICLE_CLASSES  # noqa: E402

import numpy as np  # noqa: E402
import random  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402


def detect_cloud_environment() -> str:
    if os.path.exists("/content"):
        return "colab"
    if os.path.exists("/kaggle"):
        return "kaggle"
    return "local"


def default_workspace(env_name: str) -> str:
    if env_name == "colab":
        return "/content/vehicle_training_workspace"
    if env_name == "kaggle":
        return "/kaggle/working/vehicle_training_workspace"
    return str(Path.cwd() / "vehicle_training_workspace")


def find_dataset_root(base_dir: Path) -> Path:
    if (base_dir / "images").is_dir() and (base_dir / "annotations").is_dir():
        return base_dir

    for candidate in base_dir.rglob("*"):
        if candidate.is_dir() and (candidate / "images").is_dir() and (candidate / "annotations").is_dir():
            return candidate

    raise FileNotFoundError(
        "Dataset tidak ditemukan. Harus ada folder images/ dan annotations/ di dalam dataset."
    )


def prepare_dataset(dataset_dir: str | None, dataset_zip: str | None, workspace: str) -> Path:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    if dataset_dir:
        dataset_root = find_dataset_root(Path(dataset_dir).resolve())
        print(f"[Dataset] Menggunakan folder dataset: {dataset_root}")
        return dataset_root

    if not dataset_zip:
        raise ValueError("Gunakan salah satu: --dataset-dir atau --dataset-zip")

    zip_path = Path(dataset_zip).resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"File ZIP dataset tidak ditemukan: {zip_path}")

    extract_dir = workspace_path / "extracted_dataset"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Dataset] Ekstrak ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)

    dataset_root = find_dataset_root(extract_dir)
    print(f"[Dataset] Dataset hasil ekstrak: {dataset_root}")
    return dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cloud-ready training untuk model deteksi kendaraan di Colab/Kaggle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default=None, help="Folder dataset yang berisi images/ dan annotations/")
    parser.add_argument("--dataset-zip", default=None, help="File ZIP dataset LabelImg / Pascal VOC")
    parser.add_argument("--workspace", default=None, help="Folder kerja sementara untuk ekstraksi dan proses training")
    parser.add_argument("--output", default=None, help="Folder output model/checkpoint")
    parser.add_argument("--epochs", type=int, default=40, help="Jumlah epoch training")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate awal")
    parser.add_argument("--img-size", type=int, default=320, help="Ukuran input model")
    parser.add_argument("--val-split", type=float, default=0.15, help="Proporsi data validasi")
    parser.add_argument("--workers", type=int, default=2, help="Jumlah worker DataLoader")
    parser.add_argument("--resume", default=None, help="Checkpoint .pth untuk melanjutkan training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-pretrained", action="store_true", help="Nonaktifkan pretrained backbone")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_name = detect_cloud_environment()
    workspace = args.workspace or default_workspace(env_name)
    output_dir = args.output or str(Path(workspace) / "output")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(workspace, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_root = prepare_dataset(args.dataset_dir, args.dataset_zip, workspace)
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 68)
    print("CLOUD VEHICLE DETECTOR TRAINING")
    print("=" * 68)
    print(f"Environment : {env_name}")
    print(f"Workspace   : {workspace}")
    print(f"Dataset     : {dataset_root}")
    print(f"Output      : {output_dir}")
    print(f"Device      : {device}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch       : {args.batch}")
    print(f"Classes     : {VEHICLE_CLASSES}")
    print("=" * 68)

    train_full = VehicleDataset(
        images_dir=str(images_dir),
        annotations_dir=str(annotations_dir),
        transforms=get_train_transforms(),
    )
    val_full = VehicleDataset(
        images_dir=str(images_dir),
        annotations_dir=str(annotations_dir),
        transforms=None,
    )

    total_samples = len(train_full)
    val_count = max(1, int(total_samples * args.val_split))
    train_count = total_samples - val_count
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(total_samples, generator=generator).tolist()

    train_dataset = Subset(train_full, indices[:train_count])
    val_dataset = Subset(val_full, indices[train_count:])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(NUM_CLASSES, pretrained_backbone=not args.no_pretrained)
    model.to(device)

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    milestones = [max(1, int(args.epochs * 0.6)), max(1, int(args.epochs * 0.8))]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"[Resume] Melanjutkan dari epoch {start_epoch}")

    def save_checkpoint(path: Path, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "classes": VEHICLE_CLASSES,
                "num_classes": NUM_CLASSES,
                "img_size": args.img_size,
            },
            path,
        )

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, args.epochs)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        print(
            f"Epoch [{epoch + 1:3d}/{args.epochs}] "
            f"Train: {train_loss:.4f} "
            f"Val: {val_loss:.4f} "
            f"LR: {current_lr:.2e}"
            + ("  *best" if is_best else "")
        )

        last_path = Path(output_dir) / "vehicle_detector_last.pth"
        best_path = Path(output_dir) / "vehicle_detector_best.pth"
        save_checkpoint(last_path, epoch)
        if is_best:
            save_checkpoint(best_path, epoch)

    best_path = Path(output_dir) / "vehicle_detector_best.pth"
    if best_path.is_file():
        checkpoint = torch.load(best_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    onnx_path = Path(output_dir) / "vehicle_detector.onnx"
    export_onnx(model, str(onnx_path), img_size=args.img_size)
    save_class_labels(output_dir)

    print("\n[Output files]")
    for name in ["vehicle_detector_best.pth", "vehicle_detector_last.pth", "vehicle_detector.onnx", "vehicle_classes.txt"]:
        file_path = Path(output_dir) / name
        if file_path.exists():
            print(f"  - {file_path}")

    print("\nSelesai. Model ONNX dapat dipakai untuk integrasi OpenCV DNN di C++.")


if __name__ == "__main__":
    main()
