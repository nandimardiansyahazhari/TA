#pragma once

namespace ta_lite {

// Camera and display parameters.
constexpr int kResWidth = 640;
constexpr int kResHeight = 480;

// Distance warning threshold.
constexpr double kDangerDistanceCm = 350.0;

// Camera calibration and class priors.
constexpr double kFocalLength = 1469.86;
constexpr double kRealWidthCarCm = 169.0;
constexpr double kRealWidthMotorbikeCm = 80.0;

// Detection thresholds.
constexpr float kCarConfidenceThreshold = 0.35F;
constexpr float kMotorbikeConfidenceThreshold = 0.30F;
constexpr float kNmsThreshold = 0.35F;
constexpr float kIouMatchThreshold = 0.30F;

// Minimum bbox size filter to suppress tiny noisy detections.
constexpr int kMinBoxWidthPx = 24;
constexpr int kMinBoxHeightPx = 24;

// Temporal smoothing factor (higher means more responsive, lower means smoother).
constexpr float kTemporalAlpha = 0.55F;

}  // namespace ta_lite
