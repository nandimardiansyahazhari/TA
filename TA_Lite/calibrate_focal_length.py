import cv2

def main():
    print("=== PROGRAM KALIBRASI FOCAL LENGTH CAMERA (VIDEO PLAYER MODE) ===")
    print("Cara kerja:")
    print("1. Program akan memutar video.")
    print("2. Tekan tombol [SPACE] untuk Pause / Play video.")
    print("3. Cari frame yang memiliki gambar kendaraan dengan jelas.")
    print("4. Saat video di-pause pada frame yang tepat, tekan tombol [c] (calibrate) untuk mulai menggambar bounding box.")
    print("5. Tekan [ESC] untuk keluar.")
    print("============================================\n")

    source = input("Masukkan path video (atau tekan ENTER untuk menggunakan kamera default): ").strip()
    if source == "":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Tidak bisa membuka video atau kamera.")
        return

    paused = False
    frame = None

    cv2.namedWindow("Video Player - Kalibrasi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Player - Kalibrasi", 800, 600)

    print("\n[KONTROL PLAYBACK]:")
    print("- [SPACE] : Pause / Play video")
    print("- [c]     : Pilih frame ini untuk kalibrasi (saat di-pause)")
    print("- [ESC]   : Keluar dari program")

    selected_frame = None

    while True:
        if not paused:
            ret, current_frame = cap.read()
            if not ret:
                print("Sudah mencapai akhir video. Mengulang dari awal...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = current_frame.copy()

        cv2.imshow("Video Player - Kalibrasi", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            paused = not paused
            if paused:
                print("\n[PAUSED] Video di-pause. Tekan [c] untuk kalibrasi pada frame ini, atau [SPACE] untuk lanjut.")
            else:
                print("\n[PLAYING] Video dilanjutkan.")
        elif key == ord('c') or key == ord('C'):
            if paused and frame is not None:
                selected_frame = frame.copy()
                break
            else:
                print("\n[WARNING] Pause video terlebih dahulu dengan menekan [SPACE] sebelum menekan [c]!")

    cv2.destroyAllWindows()

    if selected_frame is None:
        print("Kalibrasi dibatalkan.")
        return

    print("\n" + "="*40)
    print("[PETUNJUK DETEKSI ROI]:")
    print("1. Gunakan mouse untuk menggambar kotak (bounding box) pada kendaraan.")
    print("2. Tekan ENTER atau SPASI jika sudah sesuai.")
    print("3. Tekan C untuk membatalkan pilihan.")
    print("="*40 + "\n")

    roi = cv2.selectROI("Kalibrasi - Pilih Objek", selected_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Kalibrasi - Pilih Objek")

    x, y, w_pixel, h_pixel = roi
    if w_pixel == 0:
        print("Pilihan dibatalkan atau tidak valid.")
        return

    print(f"\nLebar objek terdeteksi: {w_pixel} piksel.")

    # Menggunakan loop untuk input yang lebih kokoh dari buffering terminal akibat penekanan tombol ENTER di OpenCV window
    w_real = None
    while w_real is None:
        try:
            val = input("Masukkan lebar asli objek dalam cm (contoh mobil: 169, motor: 80): ").strip()
            if not val:
                continue
            w_real = float(val)
        except ValueError:
            print("Input tidak valid. Harus berupa angka.")

    distance = None
    while distance is None:
        try:
            val = input("Masukkan jarak objek dari kamera saat ini dalam cm (contoh: 300): ").strip()
            if not val:
                continue
            distance = float(val)
        except ValueError:
            print("Input tidak valid. Harus berupa angka.")

    # Hitung focal length: f = (P * D) / W
    focal_length = (w_pixel * distance) / w_real
    print("\n" + "="*40)
    print(f"HASIL KALIBRASI:")
    print(f"Focal Length (f) = {focal_length:.2f}")
    print("="*40)
    print(f"\nMasukkan nilai ini ke dalam 'TA_Lite/detection_config.hpp':")
    print(f"constexpr double kFocalLength = {focal_length:.2f};")
    print("\nKemudian compile ulang program Anda dengan 'make program_video'.")

if __name__ == "__main__":
    main()
