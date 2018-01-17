#include <iostream>
#include <cmath>
#include <cassert>
#include <uWS/uWS.h>
#include "json.hpp"
#include "ukf.h"
#include "utils.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

// Read a number of float values into the given VectorXd.
void readVectorXd(istringstream &iss, VectorXd &out) {
  for (int i = 0; i < out.size(); i++) {
    iss >> out(i);
  }
}

int main() {
  uWS::Hub h;

  // Create a Kalman Filter instance
  UKF ukf(true, false);

  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;

  h.onMessage([&ukf, &estimations, &ground_truth]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (!(length && length > 2 && data[0] == '4' && data[1] == '2')) {
      return;
    }

    // Check if the message has data
    auto s = hasData(string(data));
    if (s == "") {
      string msg = "42[\"manual\",{}]";
      ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      return;
    }
    
    // Check if the event is telemetry
    auto j = json::parse(s);
    string event = j[0].get<string>();
    if (event != "telemetry") {
      return;
    }
    
    // j[1] is the data JSON object
    string sensor_measurment = j[1]["sensor_measurement"];
    
    MeasurementPackage meas_package;
    istringstream iss(sensor_measurment);

    // Read first element from the current line
    string sensor_type;
    iss >> sensor_type;

    if (sensor_type.compare("L") == 0) {
      // Lidar measurement: px, py
      meas_package.sensor_type_ = MeasurementPackage::LASER;
      meas_package.raw_measurements_ = VectorXd(2);
      readVectorXd(iss, meas_package.raw_measurements_);
    } else if (sensor_type.compare("R") == 0) {
      // Radar measurement: ro, theta, ro_dot
      meas_package.sensor_type_ = MeasurementPackage::RADAR;
      meas_package.raw_measurements_ = VectorXd(3);
      readVectorXd(iss, meas_package.raw_measurements_);
    }
    
    // Read timestamp
    iss >> meas_package.timestamp_;

    // Read groud truth: x_gt, y_gt, vx_gt, vy_gt
    VectorXd gt_values(4);
    readVectorXd(iss, gt_values);
    ground_truth.push_back(gt_values);
    
    // Call ProcessMeasurment(meas_package) for Kalman filter
    ukf.ProcessMeasurement(meas_package);

    // Get the current estimates and convert to (px, py, vx, vy) space
    VectorXd states = ukf.GetStates();
    VectorXd estimate(4);
    estimate << states(0), // px
                states(1), // py
                states(2) * cos(states(3)), // v * cos(yaw)
                states(2) * sin(states(3)); // v * sin(yaw)
    estimations.push_back(estimate);

    // Compute the running error metrics
    VectorXd RMSE = Utils::CalculateRMSE(estimations, ground_truth);

    // Assemble the message to be sent back
    json msgJson;
    msgJson["estimate_x"] = estimate(0);
    msgJson["estimate_y"] = estimate(1);
    msgJson["rmse_x"] =  RMSE(0);
    msgJson["rmse_y"] =  RMSE(1);
    msgJson["rmse_vx"] = RMSE(2);
    msgJson["rmse_vy"] = RMSE(3);
    auto msg = "42[\"estimate_marker\"," + msgJson.dump() + "]";
    std::cout << msg << std::endl;
    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
  });

  // We don't need this since we're not using HTTP but if it's removed the program doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}