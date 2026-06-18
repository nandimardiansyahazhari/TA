#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "detection_config.hpp"
#include "object_detector.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;
using ta_lite::DetectedObject;
using ta_lite::ObjectDetector;

namespace {

constexpr int kGraphWidth = 720;
constexpr int kGraphHeight = 500;
constexpr int kMaxHistory = 200;

int readRotationConfig() {
  std::ifstream f("rotation.txt");
  if (!f.good()) {
    f.open("../rotation.txt");
    if (!f.good()) {
      f.open("../../rotation.txt");
    }
  }
  int rot = 0;
  if (f.is_open()) {
    f >> rot;
  }
  return rot;
}

void writeRotationConfig(int rot) {
  std::ofstream f("rotation.txt", std::ios::trunc);
  if (f.is_open()) {
    f << rot;
    return;
  }
  f.open("../rotation.txt", std::ios::trunc);
  if (f.is_open()) {
    f << rot;
    return;
  }
  f.open("../../rotation.txt", std::ios::trunc);
  if (f.is_open()) {
    f << rot;
    return;
  }
}

void resizeKeepAspectRatio(const Mat& src, Mat& dst, Size dstSize) {
  double srcRatio = (double)src.cols / src.rows;
  double dstRatio = (double)dstSize.width / dstSize.height;
  int newW, newH;
  if (srcRatio > dstRatio) {
    newW = dstSize.width;
    newH = cvRound(newW / srcRatio);
  } else {
    newH = dstSize.height;
    newW = cvRound(newH * srcRatio);
  }
  Mat resized;
  resize(src, resized, Size(newW, newH));
  
  dst = Mat::zeros(dstSize, src.type());
  int x = (dstSize.width - newW) / 2;
  int y = (dstSize.height - newH) / 2;
  resized.copyTo(dst(Rect(x, y, newW, newH)));
}

struct TouchControl {
  double *speed;
  bool *exitRequested;
  bool fullscreen;
  int rotateMode;
};

void onMouse(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    TouchControl *ctrl = static_cast<TouchControl *>(userdata);
    int rx = x;
    int ry = y;
    if (ctrl->rotateMode == 90) {
      rx = y;
      ry = 720 - 1 - x;
    } else if (ctrl->rotateMode == 270) {
      rx = 1280 - 1 - y;
      ry = x;
    }

    // Check Rotate Button (always present at top right corner)
    bool rotateClicked = false;
    if (ctrl->rotateMode == 90 || ctrl->rotateMode == 270) {
      if (rx >= 1140 && rx <= 1240 && ry >= 40 && ry <= 100) {
        rotateClicked = true;
      }
    } else {
      if (x >= 580 && x <= 680 && y >= 40 && y <= 100) {
        rotateClicked = true;
      }
    }

    if (rotateClicked) {
      if (ctrl->rotateMode == 0) ctrl->rotateMode = 90;
      else if (ctrl->rotateMode == 90) ctrl->rotateMode = 270;
      else ctrl->rotateMode = 0;
      writeRotationConfig(ctrl->rotateMode);
      return;
    }

    if (ctrl->fullscreen) {
      if (ctrl->rotateMode == 90 || ctrl->rotateMode == 270) {
        // Landscape Fullscreen coordinates:
        // Speed Down Button: rx in [50, 250], ry in [600, 670]
        if (rx >= 50 && rx <= 250 && ry >= 600 && ry <= 670) {
          *(ctrl->speed) = max(*(ctrl->speed) - 5.0, 0.0);
        }
        // Speed Up Button: rx in [1030, 1230], ry in [600, 670]
        else if (rx >= 1030 && rx <= 1230 && ry >= 600 && ry <= 670) {
          *(ctrl->speed) = min(*(ctrl->speed) + 5.0, 120.0);
        }
        // Kembali Button: rx in [50, 250], ry in [50, 120]
        else if (rx >= 50 && rx <= 250 && ry >= 50 && ry <= 120) {
          *(ctrl->exitRequested) = true;
        }
      } else {
        // Fullscreen coordinates (Portrait):
        // Speed Down Button: x in [60, 320], y in [500, 580]
        if (x >= 60 && x <= 320 && y >= 500 && y <= 580) {
          *(ctrl->speed) = max(*(ctrl->speed) - 5.0, 0.0);
        }
        // Speed Up Button: x in [400, 660], y in [500, 580]
        else if (x >= 400 && x <= 660 && y >= 500 && y <= 580) {
          *(ctrl->speed) = min(*(ctrl->speed) + 5.0, 120.0);
        }
        // Back Button: x in [160, 560], y in [1100, 1200]
        else if (x >= 160 && x <= 560 && y >= 1100 && y <= 1200) {
          *(ctrl->exitRequested) = true;
        }
      }
    } else {
      // With Graph Mode coordinates:
      if (ctrl->rotateMode == 90 || ctrl->rotateMode == 270) {
        // Landscape with Graph:
        // Speed Down: rx in [800, 1000], ry in [600, 670]
        if (rx >= 800 && rx <= 1000 && ry >= 600 && ry <= 670) {
          *(ctrl->speed) = max(*(ctrl->speed) - 5.0, 0.0);
        }
        // Speed Up: rx in [1030, 1230], ry in [600, 670]
        else if (rx >= 1030 && rx <= 1230 && ry >= 600 && ry <= 670) {
          *(ctrl->speed) = min(*(ctrl->speed) + 5.0, 120.0);
        }
        // Back Button (KEMBALI): rx in [800, 1000], ry in [60, 130]
        else if (rx >= 800 && rx <= 1000 && ry >= 60 && ry <= 130) {
          *(ctrl->exitRequested) = true;
        }
        // Graph X (Stop) button: graphImg Rect(800,150,450,400), X at graph(680,40) scaled -> canvas(1225,182)
        else if (rx >= 1200 && rx <= 1250 && ry >= 162 && ry <= 202) {
          *(ctrl->exitRequested) = true;
        }
      } else {
        // Portrait with Graph:
        // Speed Down: x in [60, 320], y in [440, 510]
        if (x >= 60 && x <= 320 && y >= 440 && y <= 510) {
          *(ctrl->speed) = max(*(ctrl->speed) - 5.0, 0.0);
        }
        // Speed Up: x in [400, 660], y in [440, 510]
        else if (x >= 400 && x <= 660 && y >= 440 && y <= 510) {
          *(ctrl->speed) = min(*(ctrl->speed) + 5.0, 120.0);
        }
        // Back Button (KEMBALI KE MENU): x in [160, 560], y in [530, 600]
        else if (x >= 160 && x <= 560 && y >= 530 && y <= 600) {
          *(ctrl->exitRequested) = true;
        }
        // Graph X (Stop) button: graphImg Rect(0,780,720,500), X at graph(680,40) -> canvas(680,820)
        else if (x >= 655 && x <= 705 && y >= 800 && y <= 840) {
          *(ctrl->exitRequested) = true;
        }
      }
    }
  }
}

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

  if (!isFinalSummary) {
    // Exit/Back button on graph window: round button at center (kGraphWidth - 40, 40) which is (680, 40), radius 20
    circle(img, Point(kGraphWidth - 40, 40), 20, Scalar(50, 50, 200), -1); // Dark Red fill
    circle(img, Point(kGraphWidth - 40, 40), 20, Scalar(255, 255, 255), 2); // White outline
    // Draw an X symbol inside
    line(img, Point(kGraphWidth - 47, 33), Point(kGraphWidth - 33, 47), Scalar(255, 255, 255), 2, LINE_AA);
    line(img, Point(kGraphWidth - 33, 33), Point(kGraphWidth - 47, 47), Scalar(255, 255, 255), 2, LINE_AA);
  }
}

// Helper to look for file in candidate locations
string findModelFile(const string &filename, const vector<string> &candidates) {
  ifstream f(filename.c_str());
  if (f.good())
    return filename;

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
  bool fullscreen = false;
  string userModelPath = "";
  string userConfigPath = "";
  int rotateMode = readRotationConfig();

  // CLI Arguments Parsing
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--headless" || arg == "-h") {
      headless = true;
    } else if (arg == "--fullscreen" || arg == "-f") {
      fullscreen = true;
    } else if ((arg == "--rotate" || arg == "-r") && i + 1 < argc) {
      rotateMode = stoi(argv[++i]);
    } else if ((arg == "--camera" || arg == "-c") && i + 1 < argc) {
      cameraInput = argv[++i];
    } else if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
      userModelPath = argv[++i];
    } else if ((arg == "--config" || arg == "-cfg") && i + 1 < argc) {
      userConfigPath = argv[++i];
    } else if (arg == "--help") {
      cout << "Usage: " << argv[0] << " [options]\n"
           << "Options:\n"
           << "  --headless, -h      Run in headless mode (no window UI, "
              "output via terminal logs)\n"
           << "  --camera, -c <arg>  Camera index (e.g., 0, 1) or GStreamer "
              "pipeline string\n"
           << "  --rotate, -r <deg>  Rotate screen output (90 or 270)\n"
           << "  --model, -m <path>  Path to model (.onnx or .caffemodel)\n"
           << "  --config, -cfg <path> Path to configuration (.prototxt for "
              "Caffe)\n"
           << "  --help              Show this help message\n";
      return 0;
    }
  }

  // Open Video Capture
  VideoCapture cap;
  bool isDigit = !cameraInput.empty() &&
                 all_of(cameraInput.begin(), cameraInput.end(), ::isdigit);
  if (isDigit) {
    int camIdx = stoi(cameraInput);
    cout << "Opening camera index: " << camIdx << endl;
    cap.open(camIdx);
    if (!cap.isOpened() && camIdx == 0) {
      vector<int> fallbacks = {1, 2, 3, 4, 6, 8, 10};
      for (int idx : fallbacks) {
        cout << "Camera 0 failed. Trying camera index " << idx << "..." << endl;
        cap.open(idx);
        if (cap.isOpened()) {
          cout << "Successfully opened camera index: " << idx << endl;
          break;
        }
      }
    }
  } else {
    cout << "Opening camera with string source / GStreamer pipeline:\n  "
         << cameraInput << endl;
    cap.open(cameraInput, CAP_ANY);
  }


  // Retry loop: Kamera butuh waktu saat cold boot dari power station
  if (!cap.isOpened()) {
    cerr << "Camera not ready yet. Retrying up to 30 seconds..." << endl;
    if (!headless) {
      namedWindow("Optimized Collision Warning", WINDOW_NORMAL);
      resizeWindow("Optimized Collision Warning", 720, 1280);
      moveWindow("Optimized Collision Warning", 0, 0);
    }
    int retries = 0;
    const int maxRetries = 30;
    while (!cap.isOpened() && retries < maxRetries) {
      retries++;
      cerr << "  Retry " << retries << "/" << maxRetries << "..." << endl;
      if (!headless) {
        // Show waiting screen
        Mat waitImg = Mat::zeros(1280, 720, CV_8UC3);
        string msg1 = "Menginisialisasi Kamera...";
        string msg2 = "Percobaan: " + to_string(retries) + "/" + to_string(maxRetries);
        putText(waitImg, msg1, Point(80, 600), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2, LINE_AA);
        putText(waitImg, msg2, Point(80, 660), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 200, 200), 2, LINE_AA);
        imshow("Optimized Collision Warning", waitImg);
        waitKey(1000); // tunggu 1 detik sambil update layar
      } else {
        this_thread::sleep_for(chrono::seconds(1));
      }
      // Coba buka lagi semua index fallback
      if (isDigit) {
        int camIdx = stoi(cameraInput);
        cap.open(camIdx);
        if (!cap.isOpened()) {
          for (int idx : {1, 2, 3, 4, 6, 8, 10}) {
            cap.open(idx);
            if (cap.isOpened()) break;
          }
        }
      } else {
        cap.open(cameraInput, CAP_ANY);
      }
    }
    if (!cap.isOpened()) {
      cerr << "Error: Could not open camera after " << maxRetries << " retries." << endl;
      if (!headless) destroyAllWindows();
      return -1;
    }
    cerr << "Camera opened successfully after " << retries << " retries." << endl;
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
        "runs/detect/train/weights/best.onnx",
        "../runs/detect/train/weights/best.onnx",
        "../../runs/detect/train/weights/best.onnx",
        "output/yolov8n_vehicle.onnx",
        "../output/yolov8n_vehicle.onnx",
        "../../output/yolov8n_vehicle.onnx",
        "yolov8n_vehicle.onnx"};
    modelPath = findModelFile("runs/detect/train/weights/best.onnx", yoloCandidates);

    if (!modelPath.empty()) {
      cout << "Found YOLOv8 ONNX model: " << modelPath << endl;
    } else {
      vector<string> ssdCandidates = {
          "output/vehicle_detector.onnx", "../output/vehicle_detector.onnx",
          "../../output/vehicle_detector.onnx", "vehicle_detector.onnx"};
      modelPath = findModelFile("output/vehicle_detector.onnx", ssdCandidates);
      if (!modelPath.empty()) {
        cout << "Found SSDLite320 ONNX model: " << modelPath << endl;
      } else {
        vector<string> caffeCandidates = {
            "MobileNetSSD_deploy.caffemodel",
            "../MobileNetSSD_deploy.caffemodel",
            "../../MobileNetSSD_deploy.caffemodel"};
        vector<string> protoCandidates = {"MobileNetSSD_deploy.prototxt",
                                          "../MobileNetSSD_deploy.prototxt",
                                          "../../MobileNetSSD_deploy.prototxt"};
        modelPath =
            findModelFile("MobileNetSSD_deploy.caffemodel", caffeCandidates);
        configPath =
            findModelFile("MobileNetSSD_deploy.prototxt", protoCandidates);
        if (!modelPath.empty() && !configPath.empty()) {
          cout << "Falling back to Caffe MobileNetSSD:\n  Model: " << modelPath
               << "\n  Config: " << configPath << endl;
        } else {
          cerr << "Error: Could not locate any model files. Placed them in "
                  "output/ or the executable directory."
               << endl;
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

  bool exitRequested = false;
  TouchControl control = { &currentSpeedKmh, &exitRequested, fullscreen, rotateMode };

  if (!headless) {
    cout << "Press ESC to exit. Press W to speed up, S to slow down." << endl;
    namedWindow("Optimized Collision Warning", WINDOW_NORMAL);
    resizeWindow("Optimized Collision Warning", 720, 1280);
    moveWindow("Optimized Collision Warning", 0, 0);
    setMouseCallback("Optimized Collision Warning", onMouse, &control);
  } else {
    cout << "Running in HEADLESS mode. Logs will output to terminal. Press "
            "Ctrl+C to terminate."
         << endl;
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
        putTextBg(frame, label,
                  Point(detection.box.x, max(20, detection.box.y)), 0.45,
                  textColor, boxColor, 1);
      }

      if (detection.distanceCm < dangerDistanceCm) {
        closestDistance = min(closestDistance, detection.distanceCm);
        if (detection.isApproaching) {
          anyApproachingDanger = true;
        }
      }
    }

    // Warning banner is drawn directly on the canvas below (not on raw frame)

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
        cout << "[Telemetry] Disp FPS: " << fixed << setprecision(1)
             << displayFPS << " | Infer FPS: " << inferenceFPS
             << " | Speed: " << currentSpeedKmh << " km/h"
             << " | Safe Gap: " << (dangerDistanceCm / 100.0) << "m"
             << " | Detections: " << detections.size() << " | Warning: "
             << (anyApproachingDanger ? "AWAS! MENDAHULUI" : "AMAN") << endl;
      }
    }

    if (!headless) {
      // Keep control rotation in sync
      rotateMode = control.rotateMode;

      Mat canvas;
      if (rotateMode == 90 || rotateMode == 270) {
        canvas = Mat::zeros(720, 1280, CV_8UC3);
      } else {
        canvas = Mat::zeros(1280, 720, CV_8UC3);
      }

      if (fullscreen) {
        // Fullscreen Mode (No graph)
        if (rotateMode == 90 || rotateMode == 270) {
          // Landscape: resize video keeping aspect ratio to fill 1280x720 canvas
          Mat videoArea;
          resizeKeepAspectRatio(frame, videoArea, Size(1280, 720));
          videoArea.copyTo(canvas);

          string speedTxt = "SPEED: " + to_string(static_cast<int>(currentSpeedKmh)) + " km/h";
          char gapBuf[64];
          snprintf(gapBuf, sizeof(gapBuf), "SAFE GAP: %.1fm", dangerDistanceCm / 100.0);
          string gapTxt(gapBuf);

          putTextBg(canvas, speedTxt, Point(300, 640), 0.7, Scalar(255, 255, 255), Scalar(80, 80, 80), 2);
          putTextBg(canvas, gapTxt, Point(600, 640), 0.7, Scalar(255, 255, 255), Scalar(80, 80, 80), 2);

          // SPEED - Button (Red)
          rectangle(canvas, Rect(50, 600, 200, 70), Scalar(50, 50, 200), -1);
          rectangle(canvas, Rect(50, 600, 200, 70), Scalar(255, 255, 255), 2);
          putText(canvas, "SPEED -", Point(95, 645), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

          // SPEED + Button (Green)
          rectangle(canvas, Rect(1030, 600, 200, 70), Scalar(50, 180, 50), -1);
          rectangle(canvas, Rect(1030, 600, 200, 70), Scalar(255, 255, 255), 2);
          putText(canvas, "SPEED +", Point(1075, 645), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

          // Warning Banner
          string warnText = anyApproachingDanger ? "AWAS ADA KENDARAAN YANG AKAN MENDAHULUI" : "BISA MENYUSUL";
          Scalar warnBg = anyApproachingDanger ? Scalar(0, 0, 255) : Scalar(0, 180, 0);

          int baseline = 0;
          Size sz = getTextSize(warnText, FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
          Point pos(640 - sz.width / 2, 55);
          putTextBg(canvas, warnText, pos, 0.7, Scalar(255, 255, 255), warnBg, 2);

          // Telemetry info
          string statsText = "Display FPS: " + to_string(static_cast<int>(displayFPS)) + 
                             " | Infer FPS: " + to_string(static_cast<int>(inferenceFPS)) + 
                             " | Detections: " + to_string(detections.size());
          putText(canvas, statsText, Point(350, 110), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1, LINE_AA);

          // KEMBALI Button (Gray)
          rectangle(canvas, Rect(50, 50, 200, 70), Scalar(80, 80, 80), -1);
          rectangle(canvas, Rect(50, 50, 200, 70), Scalar(255, 255, 255), 2);
          putText(canvas, "KEMBALI", Point(100, 95), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

          // ROTATE Button (Gray)
          rectangle(canvas, Rect(1140, 40, 100, 60), Scalar(100, 100, 100), -1);
          rectangle(canvas, Rect(1140, 40, 100, 60), Scalar(255, 255, 255), 2);
          putText(canvas, "ROT", Point(1165, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

        } else {
          // Portrait: resize video keeping aspect ratio to fit 720x405 rect
          Mat videoArea;
          resizeKeepAspectRatio(frame, videoArea, Size(720, 405));
          videoArea.copyTo(canvas(Rect(0, 0, 720, 405)));

          string speedTxt = "SPEED: " + to_string(static_cast<int>(currentSpeedKmh)) + " km/h";
          char gapBuf[64];
          snprintf(gapBuf, sizeof(gapBuf), "SAFE GAP: %.1fm", dangerDistanceCm / 100.0);
          string gapTxt(gapBuf);

          putTextBg(canvas, speedTxt, Point(60, 440), 0.7, Scalar(255, 255, 255), Scalar(80, 80, 80), 2);
          putTextBg(canvas, gapTxt, Point(400, 440), 0.7, Scalar(255, 255, 255), Scalar(80, 80, 80), 2);

          // SPEED - Button (Red)
          rectangle(canvas, Rect(60, 500, 260, 80), Scalar(50, 50, 200), -1);
          rectangle(canvas, Rect(60, 500, 260, 80), Scalar(255, 255, 255), 3);
          putText(canvas, "SPEED -", Point(125, 550), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

          // SPEED + Button (Green)
          rectangle(canvas, Rect(400, 500, 260, 80), Scalar(50, 180, 50), -1);
          rectangle(canvas, Rect(400, 500, 260, 80), Scalar(255, 255, 255), 3);
          putText(canvas, "SPEED +", Point(465, 550), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

          // Warning Banner
          string warnText = anyApproachingDanger ? "AWAS ADA KENDARAAN YANG AKAN MENDAHULUI" : "BISA MENYUSUL";
          Scalar warnBg = anyApproachingDanger ? Scalar(0, 0, 255) : Scalar(0, 180, 0);

          int baseline = 0;
          Size sz = getTextSize(warnText, FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
          Point pos(360 - sz.width / 2, 650);
          putTextBg(canvas, warnText, pos, 0.7, Scalar(255, 255, 255), warnBg, 2);

          // Telemetry info
          string statsText = "Display FPS: " + to_string(static_cast<int>(displayFPS)) + 
                             " | Infer FPS: " + to_string(static_cast<int>(inferenceFPS)) + 
                             " | Detections: " + to_string(detections.size());
          putText(canvas, statsText, Point(60, 950), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1, LINE_AA);

          // BACK Button (Gray)
          rectangle(canvas, Rect(160, 1100, 400, 100), Scalar(80, 80, 80), -1);
          rectangle(canvas, Rect(160, 1100, 400, 100), Scalar(255, 255, 255), 3);
          putText(canvas, "KEMBALI KE MENU", Point(245, 1160), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

          // ROTATE Button (Gray)
          rectangle(canvas, Rect(580, 40, 100, 60), Scalar(100, 100, 100), -1);
          rectangle(canvas, Rect(580, 40, 100, 60), Scalar(255, 255, 255), 2);
          putText(canvas, "ROT", Point(605, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);
        }
      } else {
        // With Graph Mode (Combined Single Window)
        drawGraph(graphImg, displayFpsHistory, inferenceFpsHistory, false);

        if (rotateMode == 90 || rotateMode == 270) {
          // Landscape with Graph
          // Left: resize video keeping aspect ratio to fit 720x540 rect
          Mat videoArea;
          resizeKeepAspectRatio(frame, videoArea, Size(720, 540));
          videoArea.copyTo(canvas(Rect(50, 90, 720, 540)));

          // Right: resize graphImg to fit 450x400
          Mat resizedGraph;
          resize(graphImg, resizedGraph, Size(450, 400));
          resizedGraph.copyTo(canvas(Rect(800, 150, 450, 400)));

          // Speed and Safe Gap text
          string speedTxt = "SPEED: " + to_string(static_cast<int>(currentSpeedKmh)) + " km/h";
          char gapBuf[64];
          snprintf(gapBuf, sizeof(gapBuf), "SAFE GAP: %.1fm", dangerDistanceCm / 100.0);
          string gapTxt(gapBuf);

          putTextBg(canvas, speedTxt, Point(800, 575), 0.6, Scalar(255, 255, 255), Scalar(80, 80, 80), 2);
          putTextBg(canvas, gapTxt, Point(1030, 575), 0.6, Scalar(255, 255, 255), Scalar(80, 80, 80), 2);

          // SPEED - Button (Red)
          rectangle(canvas, Rect(800, 600, 200, 70), Scalar(50, 50, 200), -1);
          rectangle(canvas, Rect(800, 600, 200, 70), Scalar(255, 255, 255), 2);
          putText(canvas, "SPEED -", Point(845, 645), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

          // SPEED + Button (Green)
          rectangle(canvas, Rect(1030, 600, 200, 70), Scalar(50, 180, 50), -1);
          rectangle(canvas, Rect(1030, 600, 200, 70), Scalar(255, 255, 255), 2);
          putText(canvas, "SPEED +", Point(1075, 645), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

          // Warning Banner
          string warnText = anyApproachingDanger ? "AWAS ADA KENDARAAN YANG AKAN MENDAHULUI" : "BISA MENYUSUL";
          Scalar warnBg = anyApproachingDanger ? Scalar(0, 0, 255) : Scalar(0, 180, 0);

          int baseline = 0;
          Size sz = getTextSize(warnText, FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
          Point pos(410 - sz.width / 2, 50);
          putTextBg(canvas, warnText, pos, 0.7, Scalar(255, 255, 255), warnBg, 2);

          // Telemetry info
          string statsText = "Display FPS: " + to_string(static_cast<int>(displayFPS)) + 
                             " | Infer FPS: " + to_string(static_cast<int>(inferenceFPS)) + 
                             " | Detections: " + to_string(detections.size());
          putText(canvas, statsText, Point(50, 665), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1, LINE_AA);

          // BACK Button (Gray)
          rectangle(canvas, Rect(800, 60, 200, 70), Scalar(80, 80, 80), -1);
          rectangle(canvas, Rect(800, 60, 200, 70), Scalar(255, 255, 255), 2);
          putText(canvas, "KEMBALI", Point(850, 105), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

          // ROTATE Button (Gray)
          rectangle(canvas, Rect(1140, 40, 100, 60), Scalar(100, 100, 100), -1);
          rectangle(canvas, Rect(1140, 40, 100, 60), Scalar(255, 255, 255), 2);
          putText(canvas, "ROT", Point(1165, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

        } else {
          // Portrait with Graph
          // Top: resize video keeping aspect ratio to fit 720x405 rect
          Mat videoArea;
          resizeKeepAspectRatio(frame, videoArea, Size(720, 405));
          videoArea.copyTo(canvas(Rect(0, 0, 720, 405)));

          // Bottom: copy graphImg directly to Rect(0, 780, 720, 500)
          graphImg.copyTo(canvas(Rect(0, 780, 720, 500)));

          // Speed and Safe Gap text
          string speedTxt = "SPEED: " + to_string(static_cast<int>(currentSpeedKmh)) + " km/h";
          char gapBuf[64];
          snprintf(gapBuf, sizeof(gapBuf), "SAFE GAP: %.1fm", dangerDistanceCm / 100.0);
          string gapTxt(gapBuf);

          putTextBg(canvas, speedTxt, Point(60, 640), 0.7, Scalar(255, 255, 255), Scalar(80, 80, 80), 2);
          putTextBg(canvas, gapTxt, Point(400, 640), 0.7, Scalar(255, 255, 255), Scalar(80, 80, 80), 2);

          // SPEED - Button (Red)
          rectangle(canvas, Rect(60, 440, 260, 70), Scalar(50, 50, 200), -1);
          rectangle(canvas, Rect(60, 440, 260, 70), Scalar(255, 255, 255), 3);
          putText(canvas, "SPEED -", Point(125, 485), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

          // SPEED + Button (Green)
          rectangle(canvas, Rect(400, 440, 260, 70), Scalar(50, 180, 50), -1);
          rectangle(canvas, Rect(400, 440, 260, 70), Scalar(255, 255, 255), 3);
          putText(canvas, "SPEED +", Point(465, 485), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

          // BACK Button (Gray)
          rectangle(canvas, Rect(160, 530, 400, 70), Scalar(80, 80, 80), -1);
          rectangle(canvas, Rect(160, 530, 400, 70), Scalar(255, 255, 255), 3);
          putText(canvas, "KEMBALI KE MENU", Point(245, 575), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

          // Warning Banner
          string warnText = anyApproachingDanger ? "AWAS ADA KENDARAAN YANG AKAN MENDAHULUI" : "BISA MENYUSUL";
          Scalar warnBg = anyApproachingDanger ? Scalar(0, 0, 255) : Scalar(0, 180, 0);

          int baseline = 0;
          Size sz = getTextSize(warnText, FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
          Point pos(360 - sz.width / 2, 700);
          putTextBg(canvas, warnText, pos, 0.7, Scalar(255, 255, 255), warnBg, 2);

          // Telemetry info
          string statsText = "Display FPS: " + to_string(static_cast<int>(displayFPS)) + 
                             " | Infer FPS: " + to_string(static_cast<int>(inferenceFPS)) + 
                             " | Detections: " + to_string(detections.size());
          putText(canvas, statsText, Point(60, 750), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1, LINE_AA);

          // ROTATE Button (Gray)
          rectangle(canvas, Rect(580, 40, 100, 60), Scalar(100, 100, 100), -1);
          rectangle(canvas, Rect(580, 40, 100, 60), Scalar(255, 255, 255), 2);
          putText(canvas, "ROT", Point(605, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);
        }
      }

      // Display the final canvas with physical rotation if requested
      if (rotateMode == 90 || rotateMode == 270) {
        Mat rotatedCanvas;
        if (rotateMode == 90) {
          rotate(canvas, rotatedCanvas, ROTATE_90_CLOCKWISE);
        } else {
          rotate(canvas, rotatedCanvas, ROTATE_90_COUNTERCLOCKWISE);
        }
        imshow("Optimized Collision Warning", rotatedCanvas);
      } else {
        imshow("Optimized Collision Warning", canvas);
      }

      int key = waitKey(1);
      if (key == 27 || exitRequested || getWindowProperty("Optimized Collision Warning", WND_PROP_VISIBLE) < 1) {
        break;
      } else if (key == 'w' || key == 'W') {
        currentSpeedKmh = min(currentSpeedKmh + 5.0, 120.0);
      } else if (key == 's' || key == 'S') {
        currentSpeedKmh = max(currentSpeedKmh - 5.0, 0.0);
      }
    } else {
      // In headless mode, sleep tiny bit to reduce spin, or just waitKey(1)
      // equivalent to let OpenCV events update without block. Since no window
      // is open, waitKey(1) acts as a brief sleep.
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
        "performance_summary_optimized.png"};

    bool saved = false;
    for (const auto &path : savePaths) {
      if (imwrite(path, summaryImg)) {
        cout << "\nSaved performance summary graph to: " << path << endl;
        saved = true;
        break;
      }
    }
    if (!saved) {
      cerr << "\nError: Could not save performance summary graph to any "
              "candidate paths."
           << endl;
    }

    if (!headless) {
      // Re-setup mouse callback for summary screen on the same unified window
      bool summaryExit = false;
      struct SummaryControl {
        bool *exitRequested;
        int rotateMode;
      };
      
      rotateMode = readRotationConfig();
      SummaryControl sCtrl = { &summaryExit, rotateMode };
      
      setMouseCallback("Optimized Collision Warning", [](int event, int x, int y, int flags, void* userdata) {
        if (event == EVENT_LBUTTONDOWN) {
          SummaryControl *ctrl = static_cast<SummaryControl*>(userdata);
          int rx = x;
          int ry = y;
          if (ctrl->rotateMode == 90) {
            rx = y;
            ry = 720 - 1 - x;
          } else if (ctrl->rotateMode == 270) {
            rx = 1280 - 1 - y;
            ry = x;
          }

          // Check ROTATE button click
          bool rotateClicked = false;
          if (ctrl->rotateMode == 90 || ctrl->rotateMode == 270) {
            if (rx >= 1140 && rx <= 1240 && ry >= 40 && ry <= 100) {
              rotateClicked = true;
            }
          } else {
            if (x >= 580 && x <= 680 && y >= 40 && y <= 100) {
              rotateClicked = true;
            }
          }

          if (rotateClicked) {
            if (ctrl->rotateMode == 0) ctrl->rotateMode = 90;
            else if (ctrl->rotateMode == 90) ctrl->rotateMode = 270;
            else ctrl->rotateMode = 0;
            writeRotationConfig(ctrl->rotateMode);
            return;
          }

          // Check KEMBALI button click
          // Portrait KEMBALI: x in [160, 560], y in [950, 1050]
          // Landscape KEMBALI: rx in [50, 200], ry in [300, 380]
          if (ctrl->rotateMode == 90 || ctrl->rotateMode == 270) {
            if (rx >= 50 && rx <= 200 && ry >= 300 && ry <= 380) {
              *(ctrl->exitRequested) = true;
            }
          } else {
            if (x >= 160 && x <= 560 && y >= 950 && y <= 1050) {
              *(ctrl->exitRequested) = true;
            }
          }
        }
      }, &sCtrl);

      cout << "Performance Summary window opened. Touch KEMBALI or press any key to exit..." << endl;
      while (!summaryExit) {
        int currentRot = sCtrl.rotateMode;
        Mat canvas;
        if (currentRot == 90 || currentRot == 270) {
          canvas = Mat::zeros(720, 1280, CV_8UC3);
          
          // Draw Title
          putText(canvas, "PERFORMANCE SUMMARY", Point(440, 75), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 255, 255), 2, LINE_AA);

          // Draw the summary graph (originally 720x500) resized to fit on the right/center
          Mat resizedGraph;
          resize(summaryImg, resizedGraph, Size(800, 500));
          resizedGraph.copyTo(canvas(Rect(240, 150, 800, 500)));

          // KEMBALI Button
          rectangle(canvas, Rect(50, 300, 150, 80), Scalar(80, 80, 80), -1);
          rectangle(canvas, Rect(50, 300, 150, 80), Scalar(255, 255, 255), 2);
          putText(canvas, "KEMBALI", Point(75, 345), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

          // ROTATE Button
          rectangle(canvas, Rect(1140, 40, 100, 60), Scalar(100, 100, 100), -1);
          rectangle(canvas, Rect(1140, 40, 100, 60), Scalar(255, 255, 255), 2);
          putText(canvas, "ROT", Point(1165, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

        } else {
          canvas = Mat::zeros(1280, 720, CV_8UC3);

          // Draw Title
          putText(canvas, "PERFORMANCE SUMMARY", Point(160, 120), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 255, 255), 2, LINE_AA);

          // Draw graphImg directly in the center
          summaryImg.copyTo(canvas(Rect(0, 250, 720, 500)));

          // KEMBALI KE MENU Button
          rectangle(canvas, Rect(160, 950, 400, 100), Scalar(80, 80, 80), -1);
          rectangle(canvas, Rect(160, 950, 400, 100), Scalar(255, 255, 255), 3);
          putText(canvas, "KEMBALI KE MENU", Point(245, 1010), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

          // ROTATE Button
          rectangle(canvas, Rect(580, 40, 100, 60), Scalar(100, 100, 100), -1);
          rectangle(canvas, Rect(580, 40, 100, 60), Scalar(255, 255, 255), 2);
          putText(canvas, "ROT", Point(605, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);
        }

        if (currentRot == 90 || currentRot == 270) {
          Mat rotatedCanvas;
          if (currentRot == 90) {
            rotate(canvas, rotatedCanvas, ROTATE_90_CLOCKWISE);
          } else {
            rotate(canvas, rotatedCanvas, ROTATE_90_COUNTERCLOCKWISE);
          }
          imshow("Optimized Collision Warning", rotatedCanvas);
        } else {
          imshow("Optimized Collision Warning", canvas);
        }

        int key = waitKey(50);
        if (key >= 0) {
          break;
        }
      }
    }
  }

  return 0;
}
