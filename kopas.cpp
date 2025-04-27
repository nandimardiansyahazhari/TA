#include <ctime> // Untuk std::time dan std::localtime

// Fungsi untuk mendapatkan waktu saat ini dalam format HH:MM:SS
string getCurrentTime() {
    time_t now = time(nullptr);
    tm* localTime = localtime(&now);
    char buffer[9];
    strftime(buffer, sizeof(buffer), "%H:%M:%S", localTime);
    return string(buffer);
}

int main() {
    // ... (kode sebelumnya tetap sama)

    while (cap.read(frame)) {
        startTick = getTickCount();

        // Mendapatkan waktu saat ini
        string currentTime = getCurrentTime();

        // Menampilkan jam di pojok kanan atas
        putTextWithBackground(frame, "Jam: " + currentTime, Point(frame.cols - 150, 30),
                              FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), Scalar(0, 0, 0), 2);

        // ... (kode deteksi objek dan visualisasi tetap sama)

        imshow("Collision Warning System", frame);

        if (waitKey(1) == 27) break;
    }

    // ... (kode setelah loop tetap sama)
}