# TA_Lite Optimized for Raspberry Pi 5

Direktori ini berisi porting dari program deteksi dan peringatan jarak aman kendaraan (`main_optimized.cpp` dan `main_video.cpp`) yang telah disesuaikan dan dioptimalkan secara khusus untuk **Raspberry Pi 5** (Broadcom BCM2712, ARM Cortex-A76 quad-core).

## Fitur Optimasi & Kompatibilitas Raspberry Pi 5

1. **Optimasi Kompilasi ARM Cortex-A76**:
   * Kompilasi menggunakan bendera `-O3` dan `-ffast-math`.
   * Penargetan mikroarsitektur CPU dengan `-mcpu=cortex-a76` dan `-march=armv8.2-a+fp16+simd` untuk mengaktifkan instruksi NEON SIMD dan operasi FP16 presisi setengah pada hardware, meningkatkan kecepatan inferensi jaringan saraf tiruan (neural network).
   * Paralelisasi threading OpenCV DNN diatur maksimal 4 core (`cv::setNumThreads(4)`) dengan opsi deteksi OpenMP.

2. **Dukungan Kamera Fleksibel (`libcamerasrc` / GStreamer)**:
   * Mengatasi pembatasan stack kamera baru `libcamera` pada Raspberry Pi OS Bookworm.
   * Program dapat menerima argumen string pipeline GStreamer untuk menangkap video dari kamera resmi Raspberry Pi maupun USB webcam tradisional.

3. **Mode Tanpa Tampilan (Headless Mode)**:
   * Menambahkan opsi `--headless` atau `-h`. Sangat berguna saat menjalankan program melalui koneksi SSH tanpa monitor/display server terpasang.
   * Telemetri deteksi (FPS, Kecepatan, Warning) akan dicetak langsung ke terminal, dan berkas grafik performa final tetap disimpan ke disk.

4. **Portabilitas Model Berbasis Jalur Relatif**:
   * Menghilangkan path absolut kaku seperti `/home/ansyah/...`. Program akan mencari model YOLOv8 atau MobileNetSSD secara dinamis di folder kerja aktif atau folder parent.

---

## Langkah Kompilasi (Build)

Pastikan OpenCV sudah terinstal di Raspberry Pi 5 Anda (`sudo apt install libopencv-dev`).

```bash
# Buat folder build
mkdir -p build && cd build

# Buat berkas Makefile dengan CMake
cmake ..

# Kompilasi program menggunakan semua core CPU
make -j$(nproc)
```

---

## Cara Menjalankan Program

### 1. Program Deteksi Real-Time Kamera (`program_optimized`)

* **Menggunakan USB Webcam biasa (V4L2)**:
  ```bash
  ./program_optimized --camera 0
  ```

* **Menggunakan Kamera Resmi Raspberry Pi (Camera Module v2/v3/HQ) melalui GStreamer**:
  ```bash
  ./program_optimized --camera "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! appsink"
  ```

* **Menjalankan Tanpa Monitor (SSH Headless Mode)**:
  ```bash
  ./program_optimized --headless --camera 0
  ```

### 2. Program Deteksi Video Offline (`program_video`)

* **Menjalankan dengan GUI**:
  ```bash
  ./program_video jalan_raya.mp4
  ```

* **Menjalankan Tanpa Monitor (SSH Headless Mode)**:
  ```bash
  ./program_video jalan_raya.mp4 --headless
  ```

### Opsi Argumen yang Didukung:
* `--camera, -c <arg>` : Index kamera (contoh `0`) atau pipeline GStreamer.
* `--headless, -h`     : Menjalankan program tanpa GUI OpenCV.
* `--model, -m <path>`  : Memasukkan jalur file model custom (.onnx / .caffemodel).
* `--config, -cfg <path>`: Memasukkan jalur file konfigurasi custom (.prototxt).
* `--help`              : Menampilkan menu bantuan.
