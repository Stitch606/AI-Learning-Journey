#include "net.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {
ncnn::Net g_net;
std::string g_input_blob;
std::string g_output_blob;
std::vector<float> g_results;
int g_detection_count = 0;
constexpr int kInputSize = 416;
constexpr int kKeypointCount = 17;
constexpr int kOutputValues = 5 + kKeypointCount * 3;
constexpr int kMaxModelChannels = 4 + 80 + kKeypointCount * 3;
constexpr float kConfidenceThreshold = 0.30f;
constexpr float kNmsThreshold = 0.45f;

struct Candidate {
    float values[kOutputValues];
    float confidence;
};

float overlap(const Candidate& a, const Candidate& b) {
    const float left = std::max(a.values[0], b.values[0]);
    const float top = std::max(a.values[1], b.values[1]);
    const float right = std::min(a.values[2], b.values[2]);
    const float bottom = std::min(a.values[3], b.values[3]);
    const float width = std::max(0.f, right - left);
    const float height = std::max(0.f, bottom - top);
    const float intersection = width * height;
    const float areaA = std::max(0.f, a.values[2] - a.values[0]) *
                        std::max(0.f, a.values[3] - a.values[1]);
    const float areaB = std::max(0.f, b.values[2] - b.values[0]) *
                        std::max(0.f, b.values[3] - b.values[1]);
    return intersection / (areaA + areaB - intersection + 1e-6f);
}

float value_at(const ncnn::Mat& output, int channel, int candidate,
               bool channels_first) {
    return channels_first ? output.row(channel)[candidate]
                          : output.row(candidate)[channel];
}
}

extern "C" {
int init_model(const char* param_path, const char* bin_path,
               const char* input_blob, const char* output_blob) {
    if (!param_path || !bin_path || !input_blob || !output_blob) return -1;
    g_input_blob = input_blob;
    g_output_blob = output_blob;

    g_net.clear();
    g_net.opt.use_vulkan_compute = false;
    g_net.opt.num_threads = 1;
    g_net.opt.use_fp16_storage = false;
    g_net.opt.use_fp16_packed = false;
    g_net.opt.use_fp16_arithmetic = false;

    if (g_net.load_param(param_path) != 0) return -2;
    if (g_net.load_model(bin_path) != 0) {
        g_net.clear();
        return -3;
    }
    return 0;
}

int run_inference(const uint8_t* rgba, int width, int height) {
    g_results.clear();
    g_detection_count = 0;
    if (!rgba || width != kInputSize || height != kInputSize) return -1;

    ncnn::Mat input = ncnn::Mat::from_pixels(
        rgba, ncnn::Mat::PIXEL_RGBA2RGB, width, height);
    const float mean[3] = {0.f, 0.f, 0.f};
    const float norm[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    input.substract_mean_normalize(mean, norm);

    ncnn::Extractor extractor = g_net.create_extractor();
    if (extractor.input(g_input_blob.c_str(), input) != 0) return -2;

    ncnn::Mat output;
    if (extractor.extract(g_output_blob.c_str(), output) != 0) return -3;

    ncnn::Mat output2d = output.dims == 3 && output.c == 1 ? output.channel(0) : output;
    bool channels_first = false;
    int candidate_count = 0;
    int model_channels = 0;
    if (output2d.dims == 2 && output2d.h >= 56 && output2d.h <= kMaxModelChannels) {
        channels_first = true;
        model_channels = output2d.h;
        candidate_count = output2d.w;
    } else if (output2d.dims == 2 && output2d.w >= 56 && output2d.w <= kMaxModelChannels) {
        channels_first = false;
        model_channels = output2d.w;
        candidate_count = output2d.h;
    } else {
        return -4;
    }

    const int class_count = model_channels - 4 - kKeypointCount * 3;
    if (class_count < 1 || class_count > 80) return -5;
    const int keypoint_offset = 4 + class_count;

    std::vector<Candidate> candidates;
    candidates.reserve(candidate_count);

    for (int i = 0; i < candidate_count; ++i) {
        float best_score = 0.f;
        for (int c = 0; c < class_count; ++c) {
            best_score = std::max(best_score,
                value_at(output2d, 4 + c, i, channels_first));
        }
        if (!std::isfinite(best_score) || best_score < kConfidenceThreshold) continue;

        const float cx = value_at(output2d, 0, i, channels_first);
        const float cy = value_at(output2d, 1, i, channels_first);
        const float box_w = value_at(output2d, 2, i, channels_first);
        const float box_h = value_at(output2d, 3, i, channels_first);
        if (!std::isfinite(cx) || !std::isfinite(cy) ||
            !std::isfinite(box_w) || !std::isfinite(box_h) ||
            box_w <= 0.f || box_h <= 0.f) continue;

        Candidate candidate{};
        candidate.confidence = best_score;
        candidate.values[0] = std::max(0.f, cx - box_w * 0.5f);
        candidate.values[1] = std::max(0.f, cy - box_h * 0.5f);
        candidate.values[2] = std::min(static_cast<float>(kInputSize), cx + box_w * 0.5f);
        candidate.values[3] = std::min(static_cast<float>(kInputSize), cy + box_h * 0.5f);
        candidate.values[4] = best_score;

        for (int k = 0; k < kKeypointCount; ++k) {
            const int source = keypoint_offset + k * 3;
            const int target = 5 + k * 3;
            candidate.values[target] = value_at(output2d, source, i, channels_first);
            candidate.values[target + 1] = value_at(output2d, source + 1, i, channels_first);
            candidate.values[target + 2] = value_at(output2d, source + 2, i, channels_first);
            if (!std::isfinite(candidate.values[target]) ||
                !std::isfinite(candidate.values[target + 1]) ||
                !std::isfinite(candidate.values[target + 2])) {
                candidate.values[target + 2] = 0.f;
            }
        }
        candidates.push_back(candidate);
    }

    std::sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            return a.confidence > b.confidence;
        });

    std::vector<Candidate> selected;
    selected.reserve(50);
    for (const Candidate& candidate : candidates) {
        bool suppressed = false;
        for (const Candidate& kept : selected) {
            if (overlap(candidate, kept) > kNmsThreshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            selected.push_back(candidate);
            if (selected.size() >= 50) break;
        }
    }

    g_detection_count = static_cast<int>(selected.size());
    g_results.reserve(selected.size() * kOutputValues);
    for (const Candidate& candidate : selected) {
        g_results.insert(g_results.end(), candidate.values,
                         candidate.values + kOutputValues);
    }
    return g_detection_count;
}

int get_detection_count() {
    return g_detection_count;
}

void copy_detections(float* destination) {
    if (!destination || g_results.empty()) return;
    std::memcpy(destination, g_results.data(), g_results.size() * sizeof(float));
}
}
