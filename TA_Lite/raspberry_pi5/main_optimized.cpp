#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
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

constexpr int kGraphWidth = 640;
constexpr int kGraphHeight = 400;
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

void drawGraph(Mat &img, const vector<double> &displayHistory,
               const vector<double> &inferenceHistory, bool isFinalSummary) {
  img.setTo(Scalar(25, 25, 25)); // Premium dark background

  const int leftMargin = 70;
  const int rightMargin = 30;
  const int topMargin = 75;
  const int bottomMargin = 45;
  const int chartWidth = kGraphWidth - leftMargin - rightMargin;
  const int chartHeight = kGraphHeight - topMargin - bottomMargin;

  vector<double> dispPlot, inferPlot;
  if (isFinalSummary) {
    dispPlot = displayHistory;
    inferPlot = inferenceHistory;
  } else {
    size_t startIdx = (displayHistory.size() > kMaxHistory)
                          ? (displayHistory.size() - kMaxHistory)
                          : 0;
    dispPlot.assign(displayHistory.begin() + startIdx, displayHistory.end());
    size_t startIdxInfer = (inferenceHistory.size() > kMaxHistory)
                               ? (inferenceHistory.size() - kMaxHistory)
                               : 0;
    inferPlot.assign(inferenceHistory.begin() + startIdxInfer,
                     inferenceHistory.end());
  }

  double maxFps = 60.0;
  for (double val : dispPlot)
    if (val > maxFps)
      maxFps = val;
  for (double val : inferPlot)
    if (val > maxFps)
      maxFps = val;
  maxFps = ceil(maxFps / 10.0) * 10.0;

  rectangle(img, Rect(leftMargin, topMargin, chartWidth, chartHeight),
            Scalar(35, 35, 35), FILLED);
  rectangle(img, Rect(leftMargin, topMargin, chartWidth, chartHeight),
            Scalar(80, 80, 80), 1);

  int numYSteps = 5;
  double stepVal = maxFps / numYSteps;
  for (int i = 0; i <= numYSteps; ++i) {
    double val = i * stepVal;
    int y = topMargin + chartHeight -
            static_cast<int>((val / maxFps) * chartHeight);
    line(img, Point(leftMargin, y), Point(kGraphWidth - rightMargin, y),
         Scalar(55, 55, 55), 1, LINE_AA);

    char labelBuf[32];
    snprintf(labelBuf, sizeof(labelBuf), "%d FPS", static_cast<int>(val));
    putText(img, labelBuf, Point(10, y + 4), FONT_HERSHEY_SIMPLEX, 0.35,
            Scalar(180, 180, 180), 1, LINE_AA);
  }

  int numXSteps = 5;
  size_t totalPoints = dispPlot.size();
  for (int i = 0; i <= numXSteps; ++i) {
    double pct = static_cast<double>(i) / numXSteps;
    int x = leftMargin + static_cast<int>(pct * chartWidth);
    line(img, Point(x, topMargin), Point(x, topMargin + chartHeight),
         Scalar(55, 55, 55), 1, LINE_AA);

    string xLabel;
    if (isFinalSummary) {
      int frameNum = static_cast<int>(
          pct * (displayHistory.empty() ? 0 : (displayHistory.size() - 1)));
      xLabel = "F" + to_string(frameNum);
    } else {
      int offset = static_cast<int>((pct - 1.0) * kMaxHistory);
      xLabel = (offset == 0) ? "Now" : to_string(offset);
    }
    putText(img, xLabel, Point(x - 15, topMargin + chartHeight + 18),
            FONT_HERSHEY_SIMPLEX, 0.35, Scalar(150, 150, 150), 1, LINE_AA);
  }

  auto getPt = [&](size_t idx, double val) {
    double xPct = (totalPoints > 1)
                      ? (static_cast<double>(idx) / (totalPoints - 1))
                      : 0.0;
    int x = leftMargin + static_cast<int>(xPct * chartWidth);
    double yPct = val / maxFps;
    int y = topMargin + chartHeight - static_cast<int>(yPct * chartHeight);
    y = std::max(topMargin, std::min(y, topMargin + chartHeight));
    return Point(x, y);
  };

  if (inferPlot.size() > 1) {
    for (size_t i = 1; i < inferPlot.size(); ++i) {
      line(img, getPt(i - 1, inferPlot[i - 1]), getPt(i, inferPlot[i]),
           Scalar(50, 100, 255), 2, LINE_AA);
    }
  }
  if (dispPlot.size() > 1) {
    for (size_t i = 1; i < dispPlot.size(); ++i) {
      line(img, getPt(i - 1, dispPlot[i - 1]), getPt(i, dispPlot[i]),
           Scalar(80, 220, 100), 2, LINE_AA);
    }
  }

  if (!inferPlot.empty()) {
    Point lastPt = getPt(inferPlot.size() - 1, inferPlot.back());
    circle(img, lastPt, 4, Scalar(50, 100, 255), FILLED, LINE_AA);
    char valBuf[16];
    snprintf(valBuf, sizeof(valBuf), "%.1f", inferPlot.back());
    line(img, Point(leftMargin, lastPt.y), lastPt, Scalar(120, 120, 120), 1,
         LINE_4);
    putText(img, valBuf, Point(lastPt.x - 45, lastPt.y - 6),
            FONT_HERSHEY_SIMPLEX, 0.35, Scalar(50, 100, 255), 1, LINE_AA);
  }
  if (!dispPlot.empty()) {
    Point lastPt = getPt(dispPlot.size() - 1, dispPlot.back());
    circle(img, lastPt, 4, Scalar(80, 220, 100), FILLED, LINE_AA);
    char valBuf[16];
    snprintf(valBuf, sizeof(valBuf), "%.1f", dispPlot.back());
    line(img, Point(leftMargin, lastPt.y), lastPt, Scalar(120, 120, 120), 1,
         LINE_4);
    putText(img, valBuf, Point(lastPt.x - 45, lastPt.y - 6),
            FONT_HERSHEY_SIMPLEX, 0.35, Scalar(80, 220, 100), 1, LINE_AA);
  }

  double curDisp = 0.0, avgDisp = 0.0, maxDisp = 0.0, minDisp = 0.0;
  double curInfer = 0.0, avgInfer = 0.0, maxInfer = 0.0, minInfer = 0.0;

  if (!displayHistory.empty()) {
    curDisp = displayHistory.back();
    maxDisp = *max_element(displayHistory.begin(), displayHistory.end());
    minDisp = *min_element(displayHistory.begin(), displayHistory.end());
    double sum = 0;
    for (double v : displayHistory)
      sum += v;
    avgDisp = sum / displayHistory.size();
  }
  if (!inferenceHistory.empty()) {
    curInfer = inferenceHistory.back();
    maxInfer = *max_element(inferenceHistory.begin(), inferenceHistory.end());
    minInfer = *min_element(inferenceHistory.begin(), inferenceHistory.end());
    double sum = 0;
    for (double v : inferenceHistory)
      sum += v;
    avgInfer = sum / inferenceHistory.size();
  }

  string title = isFinalSummary ? "PERFORMANCE SUMMARY (COMPLETE RUN)"
                                : "REAL-TIME PERFORMANCE METRICS";
  putText(img, title, Point(leftMargin, 25), FONT_HERSHEY_SIMPLEX, 0.5,
          Scalar(255, 255, 255), 1, LINE_AA);

  char statBuf[128];
  circle(img, Point(leftMargin + 10, 45), 4, Scalar(80, 220, 100), FILLED,
         LINE_AA);
  snprintf(statBuf, sizeof(statBuf),
           "Display FPS:   Cur %4.1f | Avg %4.1f | Max %4.1f | Min %4.1f",
           curDisp, avgDisp, maxDisp, minDisp);
  putText(img, statBuf, Point(leftMargin + 25, 49), FONT_HERSHEY_SIMPLEX, 0.38,
          Scalar(200, 200, 200), 1, LINE_AA);

  circle(img, Point(leftMargin + 10, 63), 4, Scalar(50, 100, 255), FILLED,
         LINE_AA);
  snprintf(statBuf, sizeof(statBuf),
           "Inference FPS: Cur %4.1f | Avg %4.1f | Max %4.1f | Min %4.1f",
           curInfer, avgInfer, maxInfer, minInfer);
  putText(img, statBuf, Point(leftMargin + 25, 67), FONT_HERSHEY_SIMPLEX, 0.38,
          Scalar(200, 200, 200), 1, LINE_AA);
}

// Helper to look for file in candidate locations
string findModelFile(const string &filename, const vector<string> &candidates) {
  ifstream f(filename.c_str());
  if (f.good()) return filename;

  for (const auto &candidate : candidates) {
    ifstream fc(candidate.c_str());
    if (fc.good()) {
      return candidate;
    }
  }
  return "";
}

} // namespace

int main(int argc, char **argv) {
  string cameraInput = "0";
  bool headless = false;
  string userModelPath = "";
  string userConfigPath = "";

  // CLI Arguments Parsing
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--headless" || arg == "-h") {
      headless = true;
    } else if ((arg == "--camera" || arg == "-c") && i + 1 < argc) {
      cameraInput = argv[++i];
    } else if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
      userModelPath = argv[++i];
    } else if ((arg == "--config" || arg == "-cfg") && i + 1 < argc) {
      userConfigPath = argv[++i];
    } else if (arg == "--help") {
      cout << "Usage: " << argv[0] << " [options]\n"
           << "Options:\n"
           << "  --headless, -h      Run in headless mode (no window UI, output via terminal logs)\n"
           << "  --camera, -c <arg>  Camera index (e.g., 0, 1) or GStreamer pipeline string\n"
           << "  --model, -m <path>  Path to model (.onnx or .caffemodel)\n"
           << "  --config, -cfg <path> Path to configuration (.prototxt for Caffe)\n"
           << "  --help              Show this help message\n";
      return 0;
    }
  }

  // Open Video Capture
  VideoCapture cap;
  bool isDigit = !cameraInput.empty() && all_of(cameraInput.begin(), cameraInput.end(), ::isdigit);
  if (isDigit) {
    int camIdx = stoi(cameraInput);
    cout << "Opening camera index: " << camIdx << endl;
    cap.open(camIdx);
    if (!cap.isOpened() && camIdx == 0) {
      cout << "Camera 0 failed. Trying camera 1..." << endl;
      cap.open(1);
    }
  } else {
    cout << "Opening camera with string source / GStreamer pipeline:\n  " << cameraInput << endl;
    cap.open(cameraInput, CAP_ANY);
  }

  if (!cap.isOpened()) {
    cerr << "Error: Could not open camera source." << endl;
    return -1;
  }

  cap.set(CAP_PROP_FRAME_WIDTH, ta_lite::kResWidth);
  cap.set(CAP_PROP_FRAME_HEIGHT, ta_lite::kResHeight);

  // Determine model paths using relative smart fallbacks
  string modelPath = "";
  string configPath = "";

  if (!userModelPath.empty()) {
    modelPath = userModelPath;
    configPath = userConfigPath;
  } else {
    // Try to find the YOLOv8 model in workspace candidate locations
    vector<string> yoloCandidates = {
      "output/yolov8n_vehicle.onnx",
      "../output/yolov8n_vehicle.onnx",
      "../../output/yolov8n_vehicle.onnx",
      "runs/detect/train/weights/best.onnx",
      "../runs/detect/train/weights/best.onnx",
      "../../runs/detect/train/weights/best.onnx",
      "yolov8n_vehicle.onnx"
    };
    modelPath = findModelFile("output/yolov8n_vehicle.onnx", yoloCandidates);

    if (!modelPath.empty()) {
      cout << "Found YOLOv8 ONNX model: " << modelPath << endl;
    } else {
      vector<string> ssdCandidates = {
        "output/vehicle_detector.onnx",
        "../output/vehicle_detector.onnx",
        "../../output/vehicle_detector.onnx",
        "vehicle_detector.onnx"
      };
      modelPath = findModelFile("output/vehicle_detector.onnx", ssdCandidates);
      if (!modelPath.empty()) {
        cout << "Found SSDLite320 ONNX model: " << modelPath << endl;
      } else {
        vector<string> caffeCandidates = {
          "MobileNetSSD_deploy.caffemodel",
          "../MobileNetSSD_deploy.caffemodel",
          "../../MobileNetSSD_deploy.caffemodel"
        };
        vector<string> protoCandidates = {
          "MobileNetSSD_deploy.prototxt",
          "../MobileNetSSD_deploy.prototxt",
          "../../MobileNetSSD_deploy.prototxt"
        };
        modelPath = findModelFile("MobileNetSSD_deploy.caffemodel", caffeCandidates);
        configPath = findModelFile("MobileNetSSD_deploy.prototxt", protoCandidates);
        if (!modelPath.empty() && !configPath.empty()) {
          cout << "Falling back to Caffe MobileNetSSD:\n  Model: " << modelPath 
               << "\n  Config: " << configPath << endl;
        } else {
          cerr << "Error: Could not locate any model files. Placed them in output/ or the executable directory." << endl;
          return -1;
        }
      }
    }
  }

  ObjectDetector detector(modelPath, configPath);
  if (!detector.isLoaded()) {
    cerr << "Error: Could not load model files." << endl;
    return -1;
  }

  Mat frame;
  Mat graphImg(kGraphHeight, kGraphWidth, CV_8UC3, Scalar(25, 25, 25));

  vector<double> displayFpsHistory;
  vector<double> inferenceFpsHistory;

  double displayFPS = 0.0;
  int frameCounter = 0;
  auto lastDisplayTick = high_resolution_clock::now();
  double currentSpeedKmh = 60.0;

  if (!headless) {
    cout << "Press ESC to exit. Press W to speed up, S to slow down." << endl;
    namedWindow("Optimized Collision Warning", WINDOW_NORMAL);
    namedWindow("Performance Metrics", WINDOW_NORMAL);
    resizeWindow("Optimized Collision Warning", 1280, 720);
    resizeWindow("Performance Metrics", 640, 400);
  } else {
    cout << "Running in HEADLESS mode. Logs will output to terminal. Press Ctrl+C to terminate." << endl;
  }

  while (true) {
    cap >> frame;
    if (frame.empty()) {
      break;
    }

    double dangerDistanceCm = 150.0 + (currentSpeedKmh * 6.0);

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
      Scalar textColor(0, 0, 0);  // Default Black
      if (detection.label == "car") {
        boxColor = Scalar(255, 0, 0);      // Blue in BGR
        textColor = Scalar(255, 255, 255); // White text for blue background
      } else if (detection.label == "motor") {
        boxColor = Scalar(0, 255, 0); // Green in BGR
        textColor = Scalar(0, 0, 0);  // Black text for green background
      }

      if (!headless) {
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
      }

      if (detection.distanceCm < dangerDistanceCm) {
        closestDistance = min(closestDistance, detection.distanceCm);
        if (detection.isApproaching) {
          anyApproachingDanger = true;
        }
      }
    }

    if (!headless) {
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

      if (headless) {
        cout << "[Telemetry] Disp FPS: " << fixed << setprecision(1) << displayFPS 
             << " | Infer FPS: " << inferenceFPS 
             << " | Speed: " << currentSpeedKmh << " km/h"
             << " | Safe Gap: " << (dangerDistanceCm / 100.0) << "m"
             << " | Detections: " << detections.size()
             << " | Warning: " << (anyApproachingDanger ? "AWAS! MENDAHULUI" : "AMAN") << endl;
      }
    }

    if (!headless) {
      drawGraph(graphImg, displayFpsHistory, inferenceFpsHistory, false);

      putTextBg(frame, "Display FPS: " + to_string(static_cast<int>(displayFPS)),
                Point(10, frame.rows - 38), 0.6, Scalar(0, 255, 0),
                Scalar(0, 0, 0), 1);
      putTextBg(frame,
                "Inference FPS: " + to_string(static_cast<int>(inferenceFPS)),
                Point(10, frame.rows - 12), 0.6, Scalar(255, 80, 30),
                Scalar(0, 0, 0), 1);

      string speedText = "Speed: " + to_string(static_cast<int>(currentSpeedKmh)) + " km/h (W/S)";
      char safeDistBuf[64];
      snprintf(safeDistBuf, sizeof(safeDistBuf), "Safe Gap: %.1fm", dangerDistanceCm / 100.0);
      string safeText(safeDistBuf);

      putTextBg(frame, speedText, Point(frame.cols - 240, frame.rows - 38), 0.6, Scalar(255, 255, 255),
                Scalar(120, 40, 40), 1);
      putTextBg(frame, safeText, Point(frame.cols - 240, frame.rows - 12), 0.6, Scalar(255, 255, 255),
                Scalar(120, 40, 40), 1);

      imshow("Optimized Collision Warning", frame);
      imshow("Performance Metrics", graphImg);

      int key = waitKey(1);
      if (key == 27) {
        break;
      } else if (key == 'w' || key == 'W') {
        currentSpeedKmh = min(currentSpeedKmh + 5.0, 120.0);
      } else if (key == 's' || key == 'S') {
        currentSpeedKmh = max(currentSpeedKmh - 5.0, 0.0);
      }
    } else {
      // In headless mode, sleep tiny bit to reduce spin, or just waitKey(1) equivalent
      // to let OpenCV events update without block.
      // Since no window is open, waitKey(1) acts as a brief sleep.
      waitKey(1);
    }
  }

  cap.release();
  if (!headless) {
    destroyAllWindows();
  }

  if (!displayFpsHistory.empty()) {
    Mat summaryImg(kGraphHeight, kGraphWidth, CV_8UC3);
    drawGraph(summaryImg, displayFpsHistory, inferenceFpsHistory, true);

    // Save image with fallback paths
    vector<string> savePaths = {
      "output/performance_summary_optimized.png",
      "../output/performance_summary_optimized.png",
      "../../output/performance_summary_optimized.png",
      "performance_summary_optimized.png"
    };

    bool saved = false;
    for (const auto &path : savePaths) {
      if (imwrite(path, summaryImg)) {
        cout << "\nSaved performance summary graph to: " << path << endl;
        saved = true;
        break;
      }
    }
    if (!saved) {
      cerr << "\nError: Could not save performance summary graph to any candidate paths." << endl;
    }

    if (!headless) {
      namedWindow("Performance Summary", WINDOW_NORMAL);
      resizeWindow("Performance Summary", 640, 400);
      imshow("Performance Summary", summaryImg);
      cout << "Performance Summary window opened. Press any key to exit..." << endl;
      waitKey(0);
    }
  }

  return 0;
}
