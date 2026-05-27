#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <deque>
#include <iostream>
#include <string>
#include <vector>

#include "detection_config.hpp"
#include "object_detector.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;
using ta_lite::DetectedObject;
using ta_lite::ObjectDetector;

namespace {

constexpr int kGraphWidth = 600;
constexpr int kGraphHeight = 300;
constexpr int kMaxHistory = 200;

void putTextBg(Mat &img, const string &text, const Point &pos, double scale,
               const Scalar &color, const Scalar &bg, int thickness = 2) {
  int baseline = 0;
  const Size textSize =
      getTextSize(text, FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
  const Rect bgRect(pos.x, pos.y - textSize.height - 4, textSize.width + 6,
                    textSize.height + baseline + 6);
  rectangle(img, bgRect, bg, FILLED);
  putText(img, text, Point(pos.x + 3, pos.y), FONT_HERSHEY_SIMPLEX, scale,
          color, thickness);
}

void drawGraph(Mat &img, const deque<double> &displayHistory,
               const deque<double> &inferenceHistory) {
  img.setTo(Scalar(242, 242, 242));

  for (int y = 0; y < kGraphHeight; y += 50) {
    line(img, Point(0, y), Point(kGraphWidth, y), Scalar(210, 210, 210), 1);
    int val = static_cast<int>(60 - (y * 60.0 / kGraphHeight));
    putText(img, to_string(val), Point(5, y + 15), FONT_HERSHEY_SIMPLEX, 0.4,
            Scalar(150, 150, 150), 1);
  }

  const double maxVal = 60.0;
  auto mapY = [maxVal](double val) {
    double y = kGraphHeight - (val / maxVal) * kGraphHeight;
    y = std::max(0.0, std::min(y, static_cast<double>(kGraphHeight - 1)));
    return static_cast<int>(y);
  };

  for (size_t i = 1; i < inferenceHistory.size(); ++i) {
    line(img,
         Point(static_cast<int>((i - 1) * kGraphWidth / kMaxHistory),
               mapY(inferenceHistory[i - 1])),
         Point(static_cast<int>(i * kGraphWidth / kMaxHistory),
               mapY(inferenceHistory[i])),
         Scalar(255, 70, 30), 2);
  }

  for (size_t i = 1; i < displayHistory.size(); ++i) {
    line(img,
         Point(static_cast<int>((i - 1) * kGraphWidth / kMaxHistory),
               mapY(displayHistory[i - 1])),
         Point(static_cast<int>(i * kGraphWidth / kMaxHistory),
               mapY(displayHistory[i])),
         Scalar(20, 180, 40), 2);
  }

  putTextBg(img, "Display FPS", Point(kGraphWidth - 150, 28), 0.45,
            Scalar(20, 180, 40), Scalar(255, 255, 255), 1);
  putTextBg(img, "Inference FPS", Point(kGraphWidth - 150, 55), 0.45,
            Scalar(255, 70, 30), Scalar(255, 255, 255), 1);
}

} // namespace

int main() {
  VideoCapture cap(0);
  if (!cap.isOpened()) {
    cap.open(1);
  }
  if (!cap.isOpened()) {
    cerr << "Error: No camera found." << endl;
    return -1;
  }

  cap.set(CAP_PROP_FRAME_WIDTH, ta_lite::kResWidth);
  cap.set(CAP_PROP_FRAME_HEIGHT, ta_lite::kResHeight);

  ObjectDetector detector(
      "/home/ansyah/TA-main/TA_Lite/MobileNetSSD_deploy.caffemodel",
      "/home/ansyah/TA-main/TA_Lite/MobileNetSSD_deploy.prototxt");
  if (!detector.isLoaded()) {
    cerr << "Error: Could not load model files." << endl;
    return -1;
  }

  Mat frame;
  Mat graphImg(kGraphHeight, kGraphWidth, CV_8UC3, Scalar(242, 242, 242));

  deque<double> displayFpsHistory(kMaxHistory, 0.0);
  deque<double> inferenceFpsHistory(kMaxHistory, 0.0);

  double displayFPS = 0.0;
  int frameCounter = 0;
  auto lastDisplayTick = high_resolution_clock::now();

  cout << "Press ESC to exit." << endl;

  namedWindow("Optimized Collision Warning", WINDOW_NORMAL);
  namedWindow("Performance Metrics", WINDOW_NORMAL);
  resizeWindow("Optimized Collision Warning", 1280, 720);
  resizeWindow("Performance Metrics", 640, 400);

  while (true) {
    cap >> frame;
    if (frame.empty()) {
      break;
    }

    auto inferStart = high_resolution_clock::now();
    const vector<DetectedObject> detections = detector.infer(frame);
    auto inferEnd = high_resolution_clock::now();

    const double inferDt =
        duration_cast<duration<double>>(inferEnd - inferStart).count();
    const double inferenceFPS = (inferDt > 0.0) ? (1.0 / inferDt) : 0.0;

    double closestDistance = 1e9;
    for (const auto &detection : detections) {
      rectangle(frame, detection.box, Scalar(0, 255, 0), 2);

      const string label =
          detection.label + " " +
          to_string(static_cast<int>(detection.distanceCm)) + "cm " +
          to_string(static_cast<int>(detection.confidence * 100.0F)) + "%";
      putTextBg(frame, label, Point(detection.box.x, max(20, detection.box.y)),
                0.45, Scalar(0, 0, 0), Scalar(0, 255, 0), 1);

      closestDistance = min(closestDistance, detection.distanceCm);
    }

    if (closestDistance < ta_lite::kDangerDistanceCm) {
      putTextBg(frame,
                "WARNING! " + to_string(static_cast<int>(closestDistance)) +
                    "cm",
                Point(frame.cols / 2 - 95, 45), 0.9, Scalar(255, 255, 255),
                Scalar(0, 0, 255), 2);
    }

    frameCounter++;
    const auto now = high_resolution_clock::now();
    const double displayDt =
        duration_cast<duration<double>>(now - lastDisplayTick).count();

    if (displayDt >= 0.1) {
      displayFPS = frameCounter / displayDt;
      frameCounter = 0;
      lastDisplayTick = now;

      displayFpsHistory.push_back(displayFPS);
      inferenceFpsHistory.push_back(inferenceFPS);
      if (displayFpsHistory.size() > kMaxHistory) {
        displayFpsHistory.pop_front();
      }
      if (inferenceFpsHistory.size() > kMaxHistory) {
        inferenceFpsHistory.pop_front();
      }
    }

    drawGraph(graphImg, displayFpsHistory, inferenceFpsHistory);

    putTextBg(frame, "Display FPS: " + to_string(static_cast<int>(displayFPS)),
              Point(10, frame.rows - 38), 0.6, Scalar(0, 255, 0),
              Scalar(0, 0, 0), 1);
    putTextBg(frame,
              "Inference FPS: " + to_string(static_cast<int>(inferenceFPS)),
              Point(10, frame.rows - 12), 0.6, Scalar(255, 80, 30),
              Scalar(0, 0, 0), 1);

    imshow("Optimized Collision Warning", frame);
    imshow("Performance Metrics", graphImg);

    if (waitKey(1) == 27) {
      break;
    }
  }

  cap.release();
  destroyAllWindows();
  return 0;
}
