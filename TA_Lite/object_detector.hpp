#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace ta_lite {

struct DetectedObject {
    std::string label;
    float confidence = 0.0F;
    cv::Rect box;
    double distanceCm = 0.0;
};

class ObjectDetector {
public:
    ObjectDetector(const std::string& modelPath, const std::string& configPath);

    bool isLoaded() const;
    std::vector<DetectedObject> infer(const cv::Mat& frame);

private:
    static float computeIoU(const cv::Rect& a, const cv::Rect& b);
    static cv::Rect smoothBox(const cv::Rect& previous, const cv::Rect& current, float alpha);
    static double smoothDistance(double previous, double current, float alpha);
    static float classThreshold(int classId);
    static double objectRealWidthCm(int classId);

    cv::dnn::Net net_;
    std::vector<std::string> classes_;
    std::vector<DetectedObject> previousDetections_;
};

}  // namespace ta_lite
