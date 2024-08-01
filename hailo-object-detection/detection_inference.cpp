#include "detection_inference.hpp"

#include <chrono>
#include <future>
#include <iostream>
#include <mutex>
#include <opencv2/core/matx.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "common.h"
#include "common/hailo_objects.hpp"
#include "common/yolo_hailortpp.hpp"
#include "hailo/hailort.hpp"

constexpr bool QUANTIZED = true;
constexpr hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO;

static std::mutex m;
static std::atomic<bool> started{false};
static MatCallback matCallback;

using namespace hailort;

static std::string info_to_str(hailo_vstream_info_t vstream_info) {
  std::string result = vstream_info.name;
  result += " (";
  result += std::to_string(vstream_info.shape.height);
  result += ", ";
  result += std::to_string(vstream_info.shape.width);
  result += ", ";
  result += std::to_string(vstream_info.shape.features);
  result += ")";
  return result;
}

// Function to generate a random color
static cv::Scalar getRandomColor() {
  cv::RNG rng(cv::getTickCount());
  int r = rng.uniform(0, 256);
  int g = rng.uniform(0, 256);
  int b = rng.uniform(0, 256);
  return cv::Scalar(r, g, b);
}

// Static map to store colors for each label across frames
static std::map<std::string, cv::Scalar> labelColors;

// Ensure this code runs once to generate colors
static void initializeColors(const std::vector<HailoDetectionPtr> &detections) {
  for (const auto &detection : detections) {
    std::string label = detection->get_label();
    if (labelColors.find(label) == labelColors.end()) {
      labelColors[label] = getRandomColor();
    }
  }
}
// Main drawing function
void drawDetections(cv::Mat &frame, const std::vector<HailoDetectionPtr> &detections, int org_width, int org_height) {
  initializeColors(detections);

  for (const auto &detection : detections) {
    if (detection->get_confidence() == 0) {
      continue;
    }

    HailoBBox bbox = detection->get_bbox();
    float xmin = bbox.xmin() * static_cast<float>(org_width);
    float ymin = bbox.ymin() * static_cast<float>(org_height);
    float xmax = bbox.xmax() * static_cast<float>(org_width);
    float ymax = bbox.ymax() * static_cast<float>(org_height);

    // Get the color for this label
    std::string label = detection->get_label();
    cv::Scalar color = labelColors[label];

    // Draw the bounding box
    cv::rectangle(frame, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax), color, 5); 

    // Prepare the label text
    std::string text = label + " " + std::to_string(static_cast<int>(detection->get_confidence() * 100)) + "%";

    // Calculate the text size
    int baseline = 0;
    float textSize = 2;
    float textThickness = 3;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, textSize, textThickness, &baseline);

    // Define the position for the text (top-left corner of the bounding box)
    cv::Point text_position(xmin, ymin - 10);

    // Create a filled rectangle as background for the text
    cv::rectangle(frame,
                  cv::Point(text_position.x, text_position.y - text_size.height - baseline),
                  cv::Point(text_position.x + text_size.width, text_position.y + baseline),
                  color,
                  cv::FILLED);

    // Add the text to the frame
    cv::putText(frame, text, text_position, cv::FONT_HERSHEY_SIMPLEX, textSize, cv::Scalar(255, 255, 255), textThickness);
  }
}

template <typename T>
static hailo_status post_processing_all(std::vector<std::shared_ptr<FeatureData<T>>> &features,
                                        std::vector<cv::Mat> &frames,
                                        double org_height,
                                        double org_width) {
  auto status = HAILO_SUCCESS;

  std::sort(features.begin(), features.end(), &FeatureData<T>::sort_tensors_by_size);

  // cv::VideoWriter video("./processed_video.mp4",
  // cv::VideoWriter::fourcc('m','p','4','v'),30, cv::Size((int)org_width,
  // (int)org_height));

  m.lock();
  std::cout << YELLOW << "\n-I- Starting postprocessing\n" << std::endl << RESET;
  m.unlock();

  while (started) {
    HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));

    for (uint j = 0; j < features.size(); j++) {
      roi->add_tensor(std::make_shared<HailoTensor>(
          reinterpret_cast<T *>(features[j]->m_buffers.get_read_buffer().data()), features[j]->m_vstream_info));
    }

    filter(roi);

    for (auto &feature : features) {
      feature->m_buffers.release_read_buffer();
    }

    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);
    cv::resize(frames[0], frames[0], cv::Size((int)org_width, (int)org_height), 1);
    drawDetections(frames[0], detections, org_width, org_height);
    // video.write(frames[0]);
    // cv::imwrite("output_image.jpg", frames[0]);

    if (matCallback != nullptr) {
      matCallback(frames[0], detections);
    }

    frames[0].release();

    m.lock();
    frames.erase(frames.begin());
    m.unlock();
  }
  // video.release();

  return status;
}

template <typename T>
static hailo_status read_all(OutputVStream &output_vstream, std::shared_ptr<FeatureData<T>> feature) {
  // m.lock();
  // std::cout << GREEN << "-I- Started read thread: " << info_to_str(output_vstream.get_info()) << std::endl <<
  // RESET; m.unlock();

  while (started) {
    std::vector<T> &buffer = feature->m_buffers.get_write_buffer();
    hailo_status status = output_vstream.read(MemoryView(buffer.data(), buffer.size()));
    feature->m_buffers.release_write_buffer();
    if (HAILO_SUCCESS != status) {
      std::cerr << "Failed reading with status = " << status << std::endl;
      return status;
    }
  }

  return HAILO_SUCCESS;
}

static hailo_status write_all(InputVStream &input_vstream, std::vector<cv::Mat> &frames) {
  m.lock();
  std::cout << CYAN << "-I- Started write thread: " << info_to_str(input_vstream.get_info()) << std::endl << RESET;
  m.unlock();

  hailo_status status = HAILO_SUCCESS;

  auto input_shape = input_vstream.get_info().shape;
  int height = input_shape.height;
  int width = input_shape.width;

  cv::VideoCapture capture;

  // get video capture device

  cv::Mat org_frame;

  while (started) {
    capture >> org_frame;
    if (org_frame.empty()) {
      break;
    }

    cv::resize(org_frame, org_frame, cv::Size(width, height), 1);
    m.lock();
    frames.push_back(org_frame);
    m.unlock();

    input_vstream.write(MemoryView(frames[frames.size() - 1].data,
                                   input_vstream.get_frame_size()));  // Writing height * width, 3
                                                                      // channels of uint8
    if (HAILO_SUCCESS != status)
      return status;

    org_frame.release();
  }

  capture.release();

  return HAILO_SUCCESS;
}

void DetectionInference::writeFrame(cv::Mat org_frame) {
  auto &input_vstream = vstreams.first[0];

  auto input_shape = input_vstream.get_info().shape;
  int height = input_shape.height;
  int width = input_shape.width;

  cv::resize(org_frame, org_frame, cv::Size(width, height), 1);
  m.lock();
  frames.push_back(org_frame);
  m.unlock();

  input_vstream.write(MemoryView(frames[frames.size() - 1].data,
                                 input_vstream.get_frame_size()));  // Writing height * width, 3
                                                                    // channels of uint8
}

template <typename T>
static hailo_status create_feature(hailo_vstream_info_t vstream_info,
                                   size_t output_frame_size,
                                   std::shared_ptr<FeatureData<T>> &feature) {
  feature = std::make_shared<FeatureData<T>>(static_cast<uint32_t>(output_frame_size),
                                             vstream_info.quant_info.qp_zp,
                                             vstream_info.quant_info.qp_scale,
                                             vstream_info.shape.width,
                                             vstream_info);

  return HAILO_SUCCESS;
}

template <typename T>
static hailo_status run_inference(std::vector<InputVStream> &input_vstream,
                                  std::vector<OutputVStream> &output_vstreams,
                                  double org_height,
                                  double org_width,
                                  std::vector<cv::Mat> &frames) {
  hailo_status status = HAILO_UNINITIALIZED;

  auto output_vstreams_size = output_vstreams.size();

  std::vector<std::shared_ptr<FeatureData<T>>> features;
  features.reserve(output_vstreams_size);
  for (size_t i = 0; i < output_vstreams_size; i++) {
    std::shared_ptr<FeatureData<T>> feature(nullptr);
    auto status = create_feature(output_vstreams[i].get_info(), output_vstreams[i].get_frame_size(), feature);
    if (HAILO_SUCCESS != status) {
      std::cerr << "Failed creating feature with status = " << status << std::endl;
      return status;
    }

    features.emplace_back(feature);
  }

  // Create the write thread
  // auto input_thread(std::async(write_all, std::ref(input_vstream[0]), std::ref(frames)));

  // Create read threads
  std::vector<std::future<hailo_status>> output_threads;
  output_threads.reserve(output_vstreams_size);
  for (size_t i = 0; i < output_vstreams_size; i++) {
    output_threads.emplace_back(std::async(read_all<T>, std::ref(output_vstreams[i]), features[i]));
  }

  // Create the postprocessing thread
  auto pp_thread(std::async(post_processing_all<T>, std::ref(features), std::ref(frames), org_height, org_width));

  for (size_t i = 0; i < output_threads.size(); i++) {
    status = output_threads[i].get();
  }
  // auto input_status = input_thread.get();
  auto pp_status = pp_thread.get();

  // if (HAILO_SUCCESS != input_status) {
  //   std::cerr << "Write thread failed with status " << input_status << std::endl;
  //   return input_status;
  // }
  if (HAILO_SUCCESS != status) {
    std::cerr << "Read failed with status " << status << std::endl;
    return status;
  }
  if (HAILO_SUCCESS != pp_status) {
    std::cerr << "Post-processing failed with status " << pp_status << std::endl;
    return pp_status;
  }

  std::cout << BOLDBLUE << "\n-I- Inference finished successfully" << RESET << std::endl;

  status = HAILO_SUCCESS;
  return status;
}

static void print_net_banner(
    std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> &vstreams) {
  std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
  std::cout << BOLDMAGENTA << "-I-  Network  Name                                     " << std::endl << RESET;
  std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
  for (auto const &value : vstreams.first) {
    std::cout << MAGENTA << "-I-  IN:  " << value.name() << std::endl << RESET;
  }
  std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
  for (auto const &value : vstreams.second) {
    std::cout << MAGENTA << "-I-  OUT: " << value.name() << std::endl << RESET;
  }
  std::cout << BOLDMAGENTA << "-I-----------------------------------------------\n" << std::endl << RESET;
}

static Expected<std::shared_ptr<ConfiguredNetworkGroup>> configure_network_group(VDevice &vdevice,
                                                                                 std::string yolo_hef) {
  auto hef_exp = Hef::create(yolo_hef);
  if (!hef_exp) {
    return make_unexpected(hef_exp.status());
  }
  auto hef = hef_exp.release();

  auto configure_params = hef.create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
  if (!configure_params) {
    return make_unexpected(configure_params.status());
  }

  auto network_groups = vdevice.configure(hef, configure_params.value());
  if (!network_groups) {
    return make_unexpected(network_groups.status());
  }

  if (1 != network_groups->size()) {
    std::cerr << "Invalid amount of network groups" << std::endl;
    return make_unexpected(HAILO_INTERNAL_FAILURE);
  }

  return std::move(network_groups->at(0));
}

std::string getCmdOption(int argc, char *argv[], const std::string &option) {
  std::string cmd;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (0 == arg.find(option, 0)) {
      std::size_t found = arg.find("=", 0) + 1;
      cmd = arg.substr(found);
      return cmd;
    }
  }
  return cmd;
}

void DetectionInference::setMatCallback(MatCallback callback) {
  matCallback = std::move(callback);
}

void DetectionInference::stop() {
  // Signal all threads to stop
  started = false;
}
int DetectionInference::startDetection(double org_height, double org_width) {
  hailo_status status = HAILO_UNINITIALIZED;
  started = true;

  std::chrono::duration<double> total_time;
  std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();

  std::string yolo_hef = "/home/sam/hailo-rpi5-examples/resources/yolov8s_h8l.hef";

  auto vdevice_exp = VDevice::create();
  if (!vdevice_exp) {
    std::cerr << "Failed create vdevice, status = " << vdevice_exp.status() << std::endl;
    return vdevice_exp.status();
  }
  auto vdevice = vdevice_exp.release();

  auto network_group_exp = configure_network_group(*vdevice, yolo_hef);
  if (!network_group_exp) {
    std::cerr << "Failed to configure network group " << yolo_hef << std::endl;
    return network_group_exp.status();
  }
  auto network_group = network_group_exp.release();

  auto vstreams_exp = VStreamsBuilder::create_vstreams(*network_group, QUANTIZED, FORMAT_TYPE);
  if (!vstreams_exp) {
    std::cerr << "Failed creating vstreams " << vstreams_exp.status() << std::endl;
    return vstreams_exp.status();
  }
  this->vstreams = vstreams_exp.release();

  print_net_banner(vstreams);

  status = run_inference<uint8_t>(std::ref(vstreams.first), std::ref(vstreams.second), org_height, org_width, frames);

  if (HAILO_SUCCESS != status) {
    std::cerr << "Failed running inference with status = " << status << std::endl;
    return status;
  }

  std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
  total_time = t_end - t_start;

  std::cout << BOLDBLUE << "\n-I- Application run finished successfully" << RESET << std::endl;
  std::cout << BOLDBLUE << "-I- Total application run time: " << (double)total_time.count() << " sec" << RESET
            << std::endl;

  // Wait for all threads to finish
  // You might need to join any std::thread objects you've created

  // Clear any remaining frames
  {
    std::lock_guard<std::mutex> lock(m);
    frames.clear();
  }

  // Release VStreams
  vstreams.first.clear();
  vstreams.second.clear();

  // Release network group and VDevice
  m_network_group.reset();
  m_vdevice.reset();

  // Any other cleanup specific to your implementation
  // ...

  std::cout << "DetectionInference stopped and resources cleaned up." << std::endl;
  return HAILO_SUCCESS;
}
