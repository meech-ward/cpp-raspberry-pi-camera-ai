
cmake_minimum_required(VERSION 3.30)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_CXX_SCAN_FOR_MODULES ON)

project(Hello CXX)

# Ensure out-of-source build
if (PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed")
endif()

find_library(LIBUNWIND_LIBRARIES NAMES unwind)

set(OpenCV_DIR "/usr/lib/aarch64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)

find_package(ZLIB REQUIRED)

include(FetchContent)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)

FetchContent_MakeAvailable(json)

add_executable(Hello main.cpp)

target_compile_features(Hello PUBLIC cxx_std_23)

target_sources(Hello
  PUBLIC FILE_SET CXX_MODULES TYPE CXX_MODULES FILES 
  server.cppm
  MappedBuffer.cppm
  ThreadPool.cppm
  SimpleCam.cppm 
  CircularFrameBuffer.cppm
)

include_directories(/usr/local/include/libcamera)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(Hello
  PRIVATE
  ${LIBUNWIND_LIBRARIES}
  ${OpenCV_LIBS}
  /usr/local/lib/uSockets.a
  /usr/local/lib/aarch64-linux-gnu/libcamera.so
  /usr/local/lib/aarch64-linux-gnu/libcamera-base.so
  ZLIB::ZLIB
  nlohmann_json::nlohmann_json
  event event_pthreads
)

message("Build successful!")