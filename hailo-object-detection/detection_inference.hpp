#ifndef DETECTION_INFERENCE_HPP
#define DETECTION_INFERENCE_HPP

#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

#include "hailo/hailort.hpp"
#include "common/hailo_objects.hpp"

using MatCallback = std::function<void(cv::Mat &, std::vector<HailoDetectionPtr> &)>;

class DetectionInference {
 private:
  std::shared_ptr<hailort::VDevice> m_vdevice;
  std::shared_ptr<hailort::ConfiguredNetworkGroup> m_network_group;
  std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> vstreams;
  std::vector<cv::Mat> frames;

 public:
  int startDetection(double org_height, double org_width);
  void writeFrame(cv::Mat org_frame);
  void stop();
  void setMatCallback(MatCallback callback);
};

#endif