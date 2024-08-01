cmake_minimum_required(VERSION 3.25)
project(vstream_detection_example_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMPILE_OPTIONS -Wall -Wextra -Wconversion -O3 -fconcepts -Wno-ignored-qualifiers -Wno-extra -Wno-stringop-truncation -Wno-reorder)

find_package(Threads)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES
./*.cpp
./common/*.cpp
)

link_libraries(stdc++fs)
add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} Threads::Threads HailoRT::libhailort)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


find_package(ZLIB REQUIRED)

include_directories(/usr/include/libcamera)

target_link_libraries(${PROJECT_NAME}
  /usr/local/lib/uSockets.a
  /usr/lib/aarch64-linux-gnu/libcamera.so
  /usr/lib/aarch64-linux-gnu/libcamera-base.so
  ZLIB::ZLIB
  nlohmann_json::nlohmann_json
)

include(FetchContent)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)