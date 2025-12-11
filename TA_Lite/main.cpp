#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <chrono>

using namespace cv;
using namespace cv::dnn;
using namespace std;
using namespace std::chrono;

// Configuration for Raspberry Pi Optimization
const int SKIP_FRAMES = 3;  // Process inference only every 3rd frame
const int RES_WIDTH = 640;
const int RES_HEIGHT = 480;

void putTextWithBackground(Mat &img, const string &text, Point pos, int fontFace, double fontScale, Scalar textColor, Scalar bgColor, int thickness = 1) {
    int baseline = 0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;
    Rect bgRect(pos.x, pos.y - textSize.height, textSize.width, textSize.height + baseline);
    rectangle(img, bgRect, bgColor, FILLED);
    putText(img, text, pos, fontFace, fontScale, textColor, thickness);
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Camera 0 not found, trying Camera 1..." << endl;
        cap = VideoCapture(1);
        if (!cap.isOpened()) {
            cerr << "Gagal membuka kamera!" << endl;
            return -1;
        }
    }

    // Pi prefers 640x480 usually, but let's try to request it gently
    // If it fails, default is fine.
    cap.set(CAP_PROP_FRAME_WIDTH, RES_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, RES_HEIGHT);

    string model = "MobileNetSSD_deploy.caffemodel";
    string config = "MobileNetSSD_deploy.prototxt";
    Net net = readNetFromCaffe(config, model);

    if (net.empty()) {
        cerr << "Gagal memuat model!" << endl;
        return -1;
    }

    // Optimization: Set backend to OpenCV Default (CPU)
    // On Pi, if user installs OpenCV with OpenVINO or HAL, they can switch this.
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    double f = 1469.86; // Focal length (calibrate for your Pi Camera if needed)
    double W_real_car = 169.0;
    double W_real_motor = 80.0;
    double danger_distance = 350.0;

    vector<string> classes = {"background", "aeroplane", "bicycle", "bird", "boat",
                              "bottle", "bus", "car", "cat", "chair", "cow", 
                              "diningtable", "dog", "horse", "motorbike", "person", 
                              "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

    Mat frame;
    long frameCount = 0;
    Mat detections; // Store detections to reuse between skipped frames
    bool detectionsAvailable = false;

    // FPS Graph Variables
    double fps = 0.0;
    int64 startTick = 0;
    vector<double> fpsHistory;
    
    // Graph Config
    int graphWidth = 600;
    int graphHeight = 400;
    int maxDataPoints = 60;

    auto programStart = high_resolution_clock::now();

    while (cap.read(frame)) {
        if (frame.empty()) continue;
        
        startTick = getTickCount();
        frameCount++;
        
        // Only run heavy inference every SKIP_FRAMES
        if (frameCount % SKIP_FRAMES == 0) {
            Mat blob = blobFromImage(frame, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5));
            net.setInput(blob);
            detections = net.forward();
            detectionsAvailable = true;
        }

        // --- DRAWING LOGIC (Runs every frame for smooth UI) ---
        if (detectionsAvailable) {
            double closest_distance = 9999.0;
            Point closest_center(-1, -1);

            for (int i = 0; i < detections.size[2]; i++) {
                float confidence = detections.ptr<float>(0)[i * 7 + 2];
                if (confidence > 0.1) { 
                    int classId = static_cast<int>(detections.ptr<float>(0)[i * 7 + 1]);

                    if (classId < classes.size() && (classes[classId] == "car" || classes[classId] == "motorbike")) {
                        int left = static_cast<int>(detections.ptr<float>(0)[i * 7 + 3] * frame.cols);
                        int top = static_cast<int>(detections.ptr<float>(0)[i * 7 + 4] * frame.rows);
                        int right = static_cast<int>(detections.ptr<float>(0)[i * 7 + 5] * frame.cols);
                        int bottom = static_cast<int>(detections.ptr<float>(0)[i * 7 + 6] * frame.rows);

                        // Clamp to frame
                        left = max(0, left); top = max(0, top);
                        right = min(frame.cols, right); bottom = min(frame.rows, bottom);

                        double W_pixel = right - left;
                        if (W_pixel <= 0) continue;

                        double W_real = (classes[classId] == "car") ? W_real_car : W_real_motor;
                        double distance = (f * W_real) / W_pixel;

                        if (distance < closest_distance) {
                            closest_distance = distance;
                            closest_center = Point((left + right)/2, (top + bottom)/2);
                        }

                        rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);
                        
                        // Simplify text for performance
                        string label = classes[classId] + " " + to_string((int)distance) + "cm";
                        putText(frame, label, Point(left, top-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 2);
                    }
                }
            }
            
            // Warnings
            if (closest_distance <= danger_distance && closest_center.x != -1) {
                 putTextWithBackground(frame, "WARNING! " + to_string((int)closest_distance) + "cm", Point(10, 30), 
                                    FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0,0,255), Scalar(255,255,0), 2);
            }
        }

        // --- FPS CALCULATION & GRAPHING ---
        double elapsedTime = (getTickCount() - startTick) / getTickFrequency();
        if (elapsedTime > 0) {
            fps = 1.0 / elapsedTime;
        }
        fpsHistory.push_back(fps);
        
        // Remove old history to save memory/speed
        if (fpsHistory.size() > maxDataPoints) {
            fpsHistory.erase(fpsHistory.begin());
        }

        // Draw FPS on Main Frame
        string fpsText = "FPS: " + to_string((int)fps);
        putTextWithBackground(frame, fpsText, Point(10, frame.rows-10), 
                            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,255,255), Scalar(0,0,0), 2);

        // Create Graph Image
        Mat graphImg(graphHeight, graphWidth, CV_8UC3, Scalar(255, 255, 255));
        
        // Draw Time Label
        auto now = high_resolution_clock::now();
        double totalRunTime = duration_cast<duration<double>>(now - programStart).count();
        int hours = static_cast<int>(totalRunTime) / 3600;
        int minutes = (static_cast<int>(totalRunTime) % 3600) / 60;
        int seconds = static_cast<int>(totalRunTime) % 60;
        string timeLabel = format("Run Time: %02d:%02d:%02d", hours, minutes, seconds);
        putTextWithBackground(graphImg, timeLabel, Point(10, 30), 
                      FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,0), Scalar(255,255,255));

        // Draw Bars
        if (!fpsHistory.empty()) {
             double maxFPS = *max_element(fpsHistory.begin(), fpsHistory.end());
             if(maxFPS < 1) maxFPS = 1;
             
             int barWidth = graphWidth / maxDataPoints;
             for(size_t i = 0; i < fpsHistory.size(); i++) {
                 double val = fpsHistory[i];
                 int barHeight = static_cast<int>((val / maxFPS) * (graphHeight - 50)); // Leave some margin
                 rectangle(graphImg, 
                           Point(i * barWidth, graphHeight - barHeight),
                           Point((i + 1) * barWidth - 1, graphHeight),
                           Scalar(0, 255, 0), FILLED);
             }
             
             putTextWithBackground(graphImg, "Max FPS: " + to_string((int)maxFPS), Point(10, 60), 
                                 FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), Scalar(0,0,0));
        }

        imshow("FPS Graph", graphImg);
        imshow("Pi Collision Warning", frame);
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
