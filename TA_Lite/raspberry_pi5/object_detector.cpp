#include "object_detector.hpp"
#include "detection_config.hpp"
#include <algorithm>
#include <limits>

namespace ta_lite {

namespace {
constexpr int kCarClassId = 1;
constexpr int kMotorbikeClassId = 2;
} // namespace

ObjectDetector::ObjectDetector(const std::string &modelPath,
                               const std::string &configPath)
    : classes_{"background", "car", "motor"} {
  if (modelPath.find("yolo") != std::string::npos ||
      modelPath.find("best.onnx") != std::string::npos) {
    isYolo_ = true;
  } else {
    isYolo_ = false;
  }

  if (modelPath.find(".onnx") != std::string::npos) {
    net_ = cv::dnn::readNetFromONNX(modelPath);
  } else {
    net_ = cv::dnn::readNetFromCaffe(configPath, modelPath);
  }
  if (!net_.empty()) {
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    
    // RPi 5 CPU optimization: use all 4 high-performance Cortex-A76 cores
    cv::setNumThreads(4);
  }
}

bool ObjectDetector::isLoaded() const { return !net_.empty(); }

std::vector<DetectedObject> ObjectDetector::infer(const cv::Mat &frame) {
  if (frame.empty() || net_.empty()) {
    return {};
  }

  cv::Mat blob;
  if (isYolo_ || classes_.size() == 3) {
    blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(320, 320),
                                  cv::Scalar(0, 0, 0), true, false);
  } else {
    blob = cv::dnn::blobFromImage(frame, 0.007843, cv::Size(300, 300),
                               cv::Scalar(127.5, 127.5, 127.5), false, false);
  }
  net_.setInput(blob);

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> classIds;

  if (isYolo_) {
    cv::Mat output = net_.forward();
    if (!output.empty() && output.dims >= 3) {
      cv::Mat outMatTemp(output.size[1], output.size[2], CV_32F, output.ptr<float>());
      cv::Mat outMat;
      cv::transpose(outMatTemp, outMat);

      const int numDetections = outMat.rows;
      for (int i = 0; i < numDetections; ++i) {
        float scoreCar = outMat.at<float>(i, 4);
        float scoreMotor = outMat.at<float>(i, 5);

        float maxScore = std::max(scoreCar, scoreMotor);
        int classId = (scoreCar > scoreMotor) ? kCarClassId : kMotorbikeClassId;

        if (maxScore < classThreshold(classId)) {
          continue;
        }

        float cx = outMat.at<float>(i, 0);
        float cy = outMat.at<float>(i, 1);
        float w = outMat.at<float>(i, 2);
        float h = outMat.at<float>(i, 3);

        int left = static_cast<int>((cx - 0.5f * w) * frame.cols);
        int top = static_cast<int>((cy - 0.5f * h) * frame.rows);
        int width = static_cast<int>(w * frame.cols);
        int height = static_cast<int>(h * frame.rows);

        left = std::max(0, std::min(left, frame.cols - 1));
        top = std::max(0, std::min(top, frame.rows - 1));
        width = std::max(1, std::min(width, frame.cols - left));
        height = std::max(1, std::min(height, frame.rows - top));

        if (width < kMinBoxWidthPx || height < kMinBoxHeightPx) {
          continue;
        }

        boxes.emplace_back(left, top, width, height);
        scores.push_back(maxScore);
        classIds.push_back(classId);
      }
    }
  } else if (classes_.size() == 3) {
    std::vector<cv::Mat> outputs;
    std::vector<cv::String> outputNames = {"boxes", "scores"};
    net_.forward(outputs, outputNames);

    if (outputs.size() < 2 || outputs[0].empty() || outputs[1].empty()) {
      return {};
    }

    cv::Mat boxesMat = outputs[0];
    cv::Mat scoresMat = outputs[1];

    const int numDetections = 3234;
    const int numClasses = 3;

    const float *boxesData = reinterpret_cast<const float *>(boxesMat.data);
    const float *scoresData = reinterpret_cast<const float *>(scoresMat.data);

    for (int i = 0; i < numDetections; ++i) {
      float maxScore = 0.0f;
      int classId = -1;

      for (int c = 1; c < numClasses; ++c) {
        float score = scoresData[i * numClasses + c];
        if (score > maxScore) {
          maxScore = score;
          classId = c;
        }
      }

      if (classId != kCarClassId && classId != kMotorbikeClassId) {
        continue;
      }

      if (maxScore < classThreshold(classId)) {
        continue;
      }

      float xmin = boxesData[i * 4 + 0];
      float ymin = boxesData[i * 4 + 1];
      float xmax = boxesData[i * 4 + 2];
      float ymax = boxesData[i * 4 + 3];

      int left = static_cast<int>(xmin * (frame.cols / 320.0f));
      int top = static_cast<int>(ymin * (frame.rows / 320.0f));
      int right = static_cast<int>(xmax * (frame.cols / 320.0f));
      int bottom = static_cast<int>(ymax * (frame.rows / 320.0f));

      left = std::max(0, std::min(left, frame.cols - 1));
      top = std::max(0, std::min(top, frame.rows - 1));
      right = std::max(0, std::min(right, frame.cols));
      bottom = std::max(0, std::min(bottom, frame.rows));

      const int width = right - left;
      const int height = bottom - top;
      if (width < kMinBoxWidthPx || height < kMinBoxHeightPx) {
        continue;
      }

      boxes.emplace_back(left, top, width, height);
      scores.push_back(maxScore);
      classIds.push_back(classId);
    }
  } else {
    cv::Mat detections = net_.forward();
    const float *data = reinterpret_cast<const float *>(detections.data);
    for (int i = 0; i < detections.size[2]; ++i) {
      const float confidence = data[i * 7 + 2];
      const int classId = static_cast<int>(data[i * 7 + 1]);

      if (classId != 7 && classId != 14) {
        continue;
      }

      int mappedClassId = (classId == 7) ? kCarClassId : kMotorbikeClassId;

      if (confidence < classThreshold(mappedClassId)) {
        continue;
      }

      int left = static_cast<int>(data[i * 7 + 3] * frame.cols);
      int top = static_cast<int>(data[i * 7 + 4] * frame.rows);
      int right = static_cast<int>(data[i * 7 + 5] * frame.cols);
      int bottom = static_cast<int>(data[i * 7 + 6] * frame.rows);

      left = std::max(0, std::min(left, frame.cols - 1));
      top = std::max(0, std::min(top, frame.rows - 1));
      right = std::max(0, std::min(right, frame.cols));
      bottom = std::max(0, std::min(bottom, frame.rows));

      const int width = right - left;
      const int height = bottom - top;
      if (width < kMinBoxWidthPx || height < kMinBoxHeightPx) {
        continue;
      }

      boxes.emplace_back(left, top, width, height);
      scores.push_back(confidence);
      classIds.push_back(mappedClassId);
    }
  }

  std::vector<int> keepIndices;
  cv::dnn::NMSBoxes(boxes, scores, 0.0F, kNmsThreshold, keepIndices);

  std::vector<DetectedObject> current;
  current.reserve(keepIndices.size());

  for (int idx : keepIndices) {
    const cv::Rect &box = boxes[idx];
    const int classId = classIds[idx];
    double focalLength = kFocalLengthCar;
    if (classId == kMotorbikeClassId) {
      focalLength = kFocalLengthMotorbike;
    }
    const double realWidth = objectRealWidthCm(classId);
    const double distanceCm = (box.width > 0)
                                  ? (focalLength * realWidth / box.width)
                                  : std::numeric_limits<double>::infinity();

    DetectedObject object;
    object.label = classes_[classId];
    object.confidence = scores[idx];
    object.box = box;
    object.distanceCm = distanceCm;

    double bestIou = 0.0;
    int bestMatch = -1;
    for (int p = 0; p < static_cast<int>(previousDetections_.size()); ++p) {
      if (previousDetections_[p].label != object.label) {
        continue;
      }
      const double iou = computeIoU(previousDetections_[p].box, object.box);
      if (iou > bestIou) {
        bestIou = iou;
        bestMatch = p;
      }
    }

    if (bestMatch >= 0 && bestIou >= kIouMatchThreshold) {
      object.isApproaching = (distanceCm < previousDetections_[bestMatch].distanceCm);
      object.box = smoothBox(previousDetections_[bestMatch].box, object.box, kTemporalAlpha);
      object.distanceCm = smoothDistance(previousDetections_[bestMatch].distanceCm,
                                         object.distanceCm, kTemporalAlpha);
    } else {
      object.isApproaching = true;
    }

    current.push_back(object);
  }

  previousDetections_ = current;
  return current;
}

float ObjectDetector::computeIoU(const cv::Rect &a, const cv::Rect &b) {
  const cv::Rect inter = a & b;
  const int interArea = inter.area();
  if (interArea <= 0) {
    return 0.0F;
  }
  const int unionArea = a.area() + b.area() - interArea;
  if (unionArea <= 0) {
    return 0.0F;
  }
  return static_cast<float>(interArea) / static_cast<float>(unionArea);
}

cv::Rect ObjectDetector::smoothBox(const cv::Rect &previous,
                                   const cv::Rect &current, float alpha) {
  auto blend = [alpha](int oldValue, int newValue) {
    return static_cast<int>((1.0F - alpha) * static_cast<float>(oldValue) +
                            alpha * static_cast<float>(newValue));
  };
  return cv::Rect(blend(previous.x, current.x), blend(previous.y, current.y),
                  blend(previous.width, current.width),
                  blend(previous.height, current.height));
}

double ObjectDetector::smoothDistance(double previous, double current,
                                      float alpha) {
  return (1.0 - static_cast<double>(alpha)) * previous +
         static_cast<double>(alpha) * current;
}

float ObjectDetector::classThreshold(int classId) {
  if (classId == kCarClassId) {
    return kCarConfidenceThreshold;
  }
  if (classId == kMotorbikeClassId) {
    return kMotorbikeConfidenceThreshold;
  }
  return 1.0F;
}

double ObjectDetector::objectRealWidthCm(int classId) {
  if (classId == kCarClassId) {
    return kRealWidthCarCm;
  }
  if (classId == kMotorbikeClassId) {
    return kRealWidthMotorbikeCm;
  }
  return kRealWidthCarCm;
}

} // namespace ta_lite
