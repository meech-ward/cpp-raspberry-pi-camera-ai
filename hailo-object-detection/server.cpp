#include "server.hpp"

#include <uWebSockets/App.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#include "SimpleCam.hpp"
#include "ThreadPool.hpp"

using namespace std;

static ThreadPool pool(1);
static uWS::App *globalApp;
static std::atomic<bool> is_streaming{false};
static SimpleCam cam;

static void handleNewFrame(const std::vector<unsigned char> &buffer, const std::vector<HailoDetectionPtr> &detections) {
  auto loop = globalApp->getLoop();

  // for (auto &detection : detections) {
  //   if (detection->get_confidence() == 0) {
  //     continue;
  //   }

  //   if (detection->get_label() == "person") {
  //     std::cout << "Hey it's a person" << std::endl;
  //   }
  // }

  loop->defer([buffer]() {
    auto stringView = std::string_view((char *)buffer.data(), buffer.size());
    uWS::OpCode opCode = uWS::BINARY;
    globalApp->publish("image-stream", stringView, opCode);
  });
}

template <bool SSL>
void logRequest(uWS::HttpResponse<SSL> *res, uWS::HttpRequest *req) {
  cout << "Incoming request: " << req->getMethod() << " " << req->getUrl() << endl;
}

void startServer(int port = 3000) {
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

              .message = [](auto *ws, string_view message, uWS::OpCode opCode) { ws->subscribe("image-stream"); },
              .close =
                  [](auto *ws, int code, string_view message) {
                    cout << "Thread " << this_thread::get_id() << " disconnected" << endl;
                  }

          })
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