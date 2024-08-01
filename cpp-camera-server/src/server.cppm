module;

#include <uWebSockets/App.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

export module server;

import ThreadPool;
import SimpleCam;

using namespace std;
namespace server {

ThreadPool pool(1);
uWS::App *globalApp;

std::atomic<bool> is_streaming{false};

void handleNewFrame(const std::vector<unsigned char> &buffer) {
  auto loop = globalApp->getLoop();

  loop->defer([buffer]() {
    auto stringView = std::string_view((char *)buffer.data(), buffer.size());
    // std::cout << "publishing image" << std::endl;
    uWS::OpCode opCode = uWS::BINARY;
    globalApp->publish("image-stream", stringView, opCode);
  });
}

template <bool SSL>
void logRequest(uWS::HttpResponse<SSL> *res, uWS::HttpRequest *req) {
  auto start = chrono::high_resolution_clock::now();
  cout << "Incoming request: " << req->getMethod() << " " << req->getUrl() << endl;
}

int height = 1080;
int width = 1920;
// int height = 720;
// int width = 1280;
// int height = 480;
// int width = 640;
int quality = 70;
SimpleCam cam{width, height, quality};

export void startServer(int port = 3000) {
  /* Note that SSL is disabled unless you build with WITH_OPENSSL=1 */
  globalApp = new uWS::App();
  uWS::App &app = *globalApp;

  cam.setFrameCallback(handleNewFrame);

  app.get("/test",
          [](auto *res, auto *req) {
            nlohmann::json user = {{"name", "sam"}};
            nlohmann::json responseJson = {{"message", "just some json from somet shit"}, {"something_else", user}};
            std::string response = responseJson.dump();

            res->writeStatus("200 OK")->writeHeader("Content-Type", "application/json")->end(response);
            logRequest(res, req);
          })
      .get("/image",
           [](auto *res, auto *req) {
             //  myCam::setupCamera();
             //  auto frame = myCam::getLatestFrame();
             //  if (frame) {
             //    res->writeStatus("200 OK")
             //        ->writeHeader("Content-Type", "image/jpeg")
             //        ->end(std::string_view((char *)frame->data(), frame->size()));
             //  } else {
             // No frame available
             nlohmann::json responseJson = {{"message", "no frame"}};
             std::string response = responseJson.dump();

             res->writeStatus("200 OK")->writeHeader("Content-Type", "application/json")->end(response);
             //  }
             logRequest(res, req);
           })
      .post("/start",
            [](auto *res, auto *req) {
              pool.enqueue([]() {
                cam.start();
                cam.go();
              });
              is_streaming = true;

              nlohmann::json responseJson = {{"message", "started"}};
              std::string response = responseJson.dump();

              res->writeStatus("200 OK")->writeHeader("Content-Type", "application/json")->end(response);
              logRequest(res, req);
            })
      .post("/stop",
            [](auto *res, auto *req) {
              cam.finish();
              nlohmann::json responseJson = {{"message", "stopped"}};
              std::string response = responseJson.dump();

              res->writeStatus("200 OK")->writeHeader("Content-Type", "application/json")->end(response);
              logRequest(res, req);
            })

      .ws<string>(
          "/ws",
          {
              .compression = uWS::SHARED_COMPRESSOR,
              .maxPayloadLength = 16 * 1024 * 1024,
              .idleTimeout = 10,
              .open =
                  [](auto *ws) {
                    cout << "Thread " << this_thread::get_id() << " connected" << endl;
                    ws->subscribe("image-stream");
                  },

              .message = [](auto *ws, string_view message, uWS::OpCode opCode) { ws->subscribe("image/opencv"); },
              .close =
                  [](auto *ws, int code, string_view message) {
                    cout << "Thread " << this_thread::get_id() << " disconnected" << endl;
                  }})
      .listen(port,
              [](auto *listen_socket) {
                if (listen_socket) {
                  /* Note that us_listen_socket_t is castable to us_socket_t */
                  cout << "Thread " << this_thread::get_id() << " listening on port "
                       << us_socket_local_port(true, (struct us_socket_t *)listen_socket) << endl;
                } else {
                  cout << "Thread " << this_thread::get_id() << " failed to listen on port 3000" << endl;
                }
              })
      .run();

  cam.finish();

  cout << "Thread " << this_thread::get_id() << " server ended" << endl;
}

}  // namespace server