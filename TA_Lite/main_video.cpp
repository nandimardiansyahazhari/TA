#include "detection_config.hpp"
#include "object_detector.hpp"
#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;
using namespace std::chrono;
using namespace ta_lite;

namespace {

constexpr int kGraphWidth = 320;
constexpr int kGraphHeight = 120;
constexpr size_t kMaxHistory = 40;

void putTextBg(Mat &img, const string &text, Point org, double fontScale,
               Scalar textColor, Scalar bgColor, int thickness) {
  int baseline = 0;
  Size textSize =
      getTextSize(text, FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
  rectangle(img, org + Point(0, baseline),
            org + Point(textSize.width, -textSize.height), bgColor, FILLED);
  putText(img, text, org, FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness,
          LINE_AA);
}

void drawGraph(Mat &img, const deque<double> &displayHistory,
               const deque<double> &inferenceHistory) {
  img.setTo(Scalar(242, 242, 242));

  auto mapY = [](double val) -> int {
    double norm = min(1.0, max(0.0, val / 60.0));
    return static_cast<int>(kGraphHeight - 10 - norm * (kGraphHeight - 20));
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

int main(int argc, char **argv) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <path_to_video>" << endl;
    return -1;
  }

  string videoPath = argv[1];
  VideoCapture cap(videoPath);
  if (!cap.isOpened()) {
    cerr << "Error: Could not open video file: " << videoPath << endl;
    return -1;
  }

  cout << "Successfully opened video: " << videoPath << endl;

  std::string modelPath =
      "/home/ansyah/TA-main/TA_Lite/runs/detect/train/weights/best.onnx";
  std::string configPath = "";

  std::ifstream f(modelPath.c_str());
  if (f.good()) {
    cout << "Loading new YOLOv8 ONNX model: " << modelPath << endl;
  } else {
    modelPath = "/home/ansyah/TA-main/TA_Lite/output/vehicle_detector.onnx";
    std::ifstream f2(modelPath.c_str());
    if (f2.good()) {
      cout << "Loading SSDLite320 ONNX model: " << modelPath << endl;
    } else {
      modelPath = "/home/ansyah/TA-main/TA_Lite/MobileNetSSD_deploy.caffemodel";
      configPath = "/home/ansyah/TA-main/TA_Lite/MobileNetSSD_deploy.prototxt";
      cout << "ONNX models not found, falling back to Caffe MobileNetSSD."
           << endl;
    }
  }

  ObjectDetector detector(modelPath, configPath);
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

  namedWindow("Optimized Collision Warning (Video)", WINDOW_NORMAL);
  namedWindow("Performance Metrics", WINDOW_NORMAL);
  resizeWindow("Optimized Collision Warning (Video)", 1280, 720);
  resizeWindow("Performance Metrics", 640, 400);

  while (true) {
    cap >> frame;
    if (frame.empty()) {
      cout << "Reached end of video." << endl;
      break;
    }

    auto inferStart = high_resolution_clock::now();
    const vector<DetectedObject> detections = detector.infer(frame);
    auto inferEnd = high_resolution_clock::now();

    const double inferDt =
        duration_cast<duration<double>>(inferEnd - inferStart).count();
    const double inferenceFPS = (inferDt > 0.0) ? (1.0 / inferDt) : 0.0;

    double closestDistance = 1e9;
    bool anyApproachingDanger = false;
    for (const auto &detection : detections) {
      Scalar boxColor(0, 255, 0); // Default Green
      Scalar textColor(0, 0, 0); // Default Black
      if (detection.label == "car") {
        boxColor = Scalar(255, 0, 0); // Blue in BGR
        textColor = Scalar(255, 255, 255); // White text for blue background
      } else if (detection.label == "motor") {
        boxColor = Scalar(0, 255, 0); // Green in BGR
        textColor = Scalar(0, 0, 0); // Black text for green background
      }

      rectangle(frame, detection.box, boxColor, 2);

      const string directionSymbol =
          detection.isApproaching ? " [->]" : " [<-]";
      const string label =
          detection.label + " " +
          to_string(static_cast<int>(detection.distanceCm)) + "cm " +
          to_string(static_cast<int>(detection.confidence * 100.0F)) + "%" +
          directionSymbol;
      putTextBg(frame, label, Point(detection.box.x, max(20, detection.box.y)),
                0.45, textColor, boxColor, 1);

      if (detection.distanceCm < ta_lite::kDangerDistanceCm) {
        closestDistance = min(closestDistance, detection.distanceCm);
        if (detection.isApproaching) {
          anyApproachingDanger = true;
        }
      }
    }

    if (anyApproachingDanger) {
      const string text = "AWAS ADA KENDARAAN YANG AKAN MENDAHULUI";
      const double scale = 0.7;
      const int thickness = 2;
      int baseline = 0;
      const Size sz =
          getTextSize(text, FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
      const Point pos(frame.cols / 2 - sz.width / 2, 45);
      putTextBg(frame, text, pos, scale, Scalar(255, 255, 255),
                Scalar(0, 0, 255), thickness);
    } else {
      const auto now = high_resolution_clock::now();
      const bool showBlink =
          (duration_cast<milliseconds>(now.time_since_epoch()).count() / 500) %
              2 ==
          0;
      if (showBlink) {
        const string text = "BISA MENYUSUL";
        const double scale = 0.7;
        const int thickness = 2;
        int baseline = 0;
        const Size sz = getTextSize(text, FONT_HERSHEY_SIMPLEX, scale,
                                    thickness, &baseline);
        const Point pos(frame.cols / 2 - sz.width / 2, 45);
        putTextBg(frame, text, pos, scale, Scalar(255, 255, 255),
                  Scalar(0, 200, 0), thickness);
      }
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

    imshow("Optimized Collision Warning (Video)", frame);
    imshow("Performance Metrics", graphImg);

    if (waitKey(1) == 27) {
      break;
    }
  }

  cap.release();
  destroyAllWindows();
  return 0;
}
