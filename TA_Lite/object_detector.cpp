#include "object_detector.hpp"

#include "detection_config.hpp"

#include <algorithm>
#include <limits>

namespace ta_lite {

namespace {

constexpr int kCarClassId = 7;
constexpr int kMotorbikeClassId = 14;

}  // namespace

ObjectDetector::ObjectDetector(const std::string& modelPath, const std::string& configPath)
    : classes_{"background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"} {
    net_ = cv::dnn::readNetFromCaffe(configPath, modelPath);
    if (!net_.empty()) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

bool ObjectDetector::isLoaded() const {
    return !net_.empty();
}

std::vector<DetectedObject> ObjectDetector::infer(const cv::Mat& frame) {
    if (frame.empty() || net_.empty()) {
        return {};
    }

    cv::Mat blob = cv::dnn::blobFromImage(
        frame, 0.007843, cv::Size(300, 300), cv::Scalar(127.5, 127.5, 127.5), false, false);
    net_.setInput(blob);
    cv::Mat detections = net_.forward();

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    const float* data = reinterpret_cast<const float*>(detections.data);
    for (int i = 0; i < detections.size[2]; ++i) {
        const float confidence = data[i * 7 + 2];
        const int classId = static_cast<int>(data[i * 7 + 1]);

        if (classId != kCarClassId && classId != kMotorbikeClassId) {
            continue;
        }

        if (confidence < classThreshold(classId)) {
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
        classIds.push_back(classId);
    }

    std::vector<int> keepIndices;
    cv::dnn::NMSBoxes(boxes, scores, 0.0F, kNmsThreshold, keepIndices);

    std::vector<DetectedObject> current;
    current.reserve(keepIndices.size());

    for (int idx : keepIndices) {
        const cv::Rect& box = boxes[idx];
        const int classId = classIds[idx];
        const double realWidth = objectRealWidthCm(classId);
        const double distanceCm = (box.width > 0) ? (kFocalLength * realWidth / box.width)
                                                  : std::numeric_limits<double>::infinity();

        DetectedObject object;
        object.label = classes_[classId];
        object.confidence = scores[idx];
        object.box = box;
        object.distanceCm = distanceCm;

        // Temporal smoothing by matching with previous frame using IoU.
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
            object.box = smoothBox(previousDetections_[bestMatch].box, object.box, kTemporalAlpha);
            object.distanceCm = smoothDistance(previousDetections_[bestMatch].distanceCm,
                                               object.distanceCm,
                                               kTemporalAlpha);
        }

        current.push_back(object);
    }

    previousDetections_ = current;
    return current;
}

float ObjectDetector::computeIoU(const cv::Rect& a, const cv::Rect& b) {
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

cv::Rect ObjectDetector::smoothBox(const cv::Rect& previous, const cv::Rect& current, float alpha) {
    auto blend = [alpha](int oldValue, int newValue) {
        return static_cast<int>((1.0F - alpha) * static_cast<float>(oldValue) +
                                alpha * static_cast<float>(newValue));
    };

    return cv::Rect(blend(previous.x, current.x),
                    blend(previous.y, current.y),
                    blend(previous.width, current.width),
                    blend(previous.height, current.height));
}

double ObjectDetector::smoothDistance(double previous, double current, float alpha) {
    return (1.0 - static_cast<double>(alpha)) * previous + static_cast<double>(alpha) * current;
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

}  // namespace ta_lite
