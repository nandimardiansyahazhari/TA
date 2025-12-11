#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <numeric>  
#include <chrono> 

using namespace cv;
using namespace cv::dnn;
using namespace std;
using namespace std::chrono;

struct FrameData {
    double fps;
    double elapsedTime; // waktu dalam detik sejak program dimulai
};
vector<FrameData> frameHistory;


void putTextWithBackground(Mat &img, const string &text, Point pos, int fontFace, double fontScale, Scalar textColor, Scalar bgColor, int thickness = 1) {
    int baseline = 0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;
    Rect bgRect(pos.x, pos.y - textSize.height, textSize.width, textSize.height + baseline);
    rectangle(img, bgRect, bgColor, FILLED);
    putText(img, text, pos, fontFace, fontScale, textColor, thickness);
}

int main() {

    auto programStart = high_resolution_clock::now();
    vector<FrameData> frameHistory;

    string model = "MobileNetSSD_deploy.caffemodel";
    string config = "MobileNetSSD_deploy.prototxt";
    Net net = readNetFromCaffe(config, model);

    if (net.empty()) {
        cerr << "Gagal memuat model!" << endl;
        return -1;
    }

  
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Camera 0 not found, trying Camera 1..." << endl;
        cap = VideoCapture(1);
        if (!cap.isOpened()) {
            cerr << "Gagal membuka kamera (0 dan 1)!" << endl;
            return -1;
        }
    }
    // Resolution settings removed to allow auto-negotiation

  
    double f = 1469.86;
    double W_real_car = 169.0;
    double W_real_motor = 80.0;

 
    double danger_distance = 350.0;
    double fps = 0.0;
    int64 startTick = 0;
    vector<double> fpsHistory;  

    vector<string> classes = {"background", "aeroplane", "bicycle", "bird", "boat",
                              "bottle", "bus", "car", "cat", "chair", "cow", 
                              "diningtable", "dog", "horse", "motorbike", "person", 
                              "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

    Mat frame;
    bool frameRead = false;
    while (cap.read(frame)) {
        if (frame.empty()) {
             cerr << "Warning: Empty frame read from camera!" << endl;
             continue;
        }
        frameRead = true;
        startTick = getTickCount();

        Mat blob = blobFromImage(frame, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5));
        net.setInput(blob);
        Mat detections = net.forward();

        double closest_distance = 9999.0;
        Point closest_center(-1, -1);

       
        for (int i = 0; i < detections.size[2]; i++) {
            float confidence = detections.ptr<float>(0)[i * 7 + 2];
            if (confidence > 0.1) {
                int classId = static_cast<int>(detections.ptr<float>(0)[i * 7 + 1]);

                if (classes[classId] == "car" || classes[classId] == "motorbike") {
                    int left = static_cast<int>(detections.ptr<float>(0)[i * 7 + 3] * frame.cols);
                    int top = static_cast<int>(detections.ptr<float>(0)[i * 7 + 4] * frame.rows);
                    int right = static_cast<int>(detections.ptr<float>(0)[i * 7 + 5] * frame.cols);
                    int bottom = static_cast<int>(detections.ptr<float>(0)[i * 7 + 6] * frame.rows);

                    
                    double W_pixel = right - left;
                    double W_real = (classes[classId] == "car") ? W_real_car : W_real_motor;
                    double distance = (f * W_real) / W_pixel;

                    if (distance < closest_distance) {
                        closest_distance = distance;
                        closest_center = Point((left + right)/2, (top + bottom)/2);
                    }

                    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);
                    stringstream distText;
                    distText << classes[classId] << ": " << fixed << setprecision(1) << distance << " cm";
                    putTextWithBackground(frame, distText.str(), Point(left, top-10), 
                                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), Scalar(0,0,0), 2);
                }
            }
        }

        if (closest_distance <= danger_distance && closest_center.x != -1) {
            string warningText = "WARNING! Jarak: " + to_string((int)closest_distance) + " cm";
            putTextWithBackground(frame, warningText, Point(10, 30), 
                                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0,0,255), Scalar(255,255,0), 3);

           
            int frameCenterX = frame.cols / 2;
            if (closest_center.x < frameCenterX - 100) {
                putTextWithBackground(frame, "Arah Bahaya: KANAN", Point(10, 70), 
                                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,255), Scalar(0,0,0), 2);
            } else if (closest_center.x > frameCenterX + 100) {
                putTextWithBackground(frame, "Arah Bahaya: KIRI", Point(10, 70), 
                                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,255), Scalar(0,0,0), 2);
            } else {
                putTextWithBackground(frame, "BAHAYA LURUS!", Point(10, 70), 
                                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,255), Scalar(0,0,0), 2);
            }
        }

        
        double elapsedTime = (getTickCount() - startTick) / getTickFrequency();
        fps = 1.0 / elapsedTime;
        fpsHistory.push_back(fps);
              
        string fpsText = "FPS: " + to_string((int)fps);
        putTextWithBackground(frame, fpsText, Point(10, frame.rows-10), 
                            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,255,255), Scalar(0,0,0), 2);

        auto now = high_resolution_clock::now();
        double totalelapsedTime = duration_cast<duration<double>>(now - programStart).count();
        frameHistory.push_back({fps, totalelapsedTime});

        int graphWidth = 600;
        int graphHeight = 400;
        Mat graphImg(graphHeight, graphWidth, CV_8UC3, Scalar(255, 255, 255));
        

        int maxDataPoints = 60;
        int startIdx = max(0, static_cast<int>(fpsHistory.size()) - maxDataPoints);
        int numPoints = min(maxDataPoints, static_cast<int>(fpsHistory.size()) - startIdx);

        int hours = static_cast<int>(elapsedTime) / 3600;
        int minutes = (static_cast<int>(elapsedTime) % 3600) / 60;
        int seconds = static_cast<int>(elapsedTime) % 60;
        int milliseconds = static_cast<int>(elapsedTime - floor(elapsedTime)) * 1000;

        string timeLabel = format("%02d:%02d:%02d.%03d", hours, minutes, seconds, milliseconds);
        putTextWithBackground(graphImg, timeLabel, Point(10, 30), 
                      FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,0), Scalar(255,255,255));


    
        
        if(numPoints > 0) {
            double maxFPS = *max_element(fpsHistory.begin() + startIdx, fpsHistory.end());
            if(maxFPS < 1) maxFPS = 1;
            
            int barWidth = graphWidth / numPoints;
            for(int i = 0; i < numPoints; i++) {
                double fpsValue = fpsHistory[startIdx + i];
                int barHeight = static_cast<int>((fpsValue / maxFPS) * graphHeight);
                rectangle(graphImg, 
                          Point(i * barWidth, graphHeight - barHeight),
                          Point((i + 1) * barWidth - 2, graphHeight),
                          Scalar(0, 255, 0), FILLED);

            }
            putTextWithBackground(graphImg, "Real-time FPS", Point(10, 30), 
                                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255), Scalar(0,0,0));
            putTextWithBackground(graphImg, "Max: " + to_string((int)maxFPS), Point(10, 60), 
                                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), Scalar(0,0,0));
        }
        
        imshow("FPS Graph", graphImg);
        imshow("Collision Warning System", frame);
        
        if (waitKey(1) == 27) break;
    }

    cap.release();
    if (frameRead) {
        destroyWindow("Collision Warning System");
    } 

if (!frameHistory.empty()) {
    double maxFPS = 0.0;
    double totalFPS = 0.0;
    double minTime = frameHistory.front().elapsedTime;
    double maxTime = frameHistory.back().elapsedTime;

    int finalWidth = 800;
    int finalHeight = 600;

    for (const auto& data : frameHistory) {
        if (data.fps > maxFPS) maxFPS = data.fps;
        totalFPS += data.fps;
    }

    Mat finalGraph(finalHeight, finalWidth, CV_8UC3, Scalar(0, 0, 0));

    // Gambar setiap titik data dengan waktu
    for (const auto& data : frameHistory) {
        double x = ((data.elapsedTime - minTime) / (maxTime - minTime)) * finalWidth;
        int y = finalHeight - (data.fps / maxFPS * finalHeight);
        circle(finalGraph, Point(x, y), 2, Scalar(0, 255, 0), FILLED);
    }

    // Tambahkan label waktu pada sumbu X
    int numTicks = 10;
    for (int i = 0; i <= numTicks; i++) {
        double t = minTime + (i * (maxTime - minTime) / numTicks);
        int xPos = (i * finalWidth) / numTicks;

        int hours = static_cast<int>(t) / 3600;
        int minutes = (static_cast<int>(t) % 3600) / 60;
        int seconds = static_cast<int>(t) % 60;
        string label = format("%02d:%02d:%02d", hours, minutes, seconds);

        putTextWithBackground(finalGraph, label, Point(xPos - 40, finalHeight - 20),
                             FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), Scalar(0,0,0));
    }

    imshow("Final Analysis", finalGraph);
    waitKey(0);
}
    destroyAllWindows();
    return 0;
}
