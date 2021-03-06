#include "ukf.h"

#include <iostream>
#include "Eigen/Dense"
#include "utils.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

UKF::UKF(bool use_laser, bool use_radar): use_laser_(use_laser),
                                          use_radar_(use_radar),
                                          is_initialized_(false),
                                          n_x_(5),
                                          n_aug_(n_x_ + 2),
                                          lambda_(3 - n_aug_) {
  // State vector
  x_ = VectorXd(n_x_);

  // Process covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Initialize sigma point weights
  Utils::GetWeights(n_aug_, lambda_, /*output*/ weights_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // This needs to be tuned.
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // This needs to be tuned.
  std_yawdd_ = 1;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
}

UKF::~UKF() {}

void UKF::InitializeMeasurement(const MeasurementPackage& m) {
  float px, py;
  float std_px, std_py;
  switch (m.sensor_type_) {
    case MeasurementPackage::RADAR: {
      // Convert radar from polar to cartesian coordinates.
      // Ignore velocity informaiton.
      px = m.raw_measurements_[0] * sin(m.raw_measurements_[1]);
      py = m.raw_measurements_[0] * cos(m.raw_measurements_[1]);
      std_px = std_py = 1;
      break;
    }
    case MeasurementPackage::LASER: {
      // Laser has position (px, py) but no velocity information.
      // Position is more acurate than radar.
      px = m.raw_measurements_[0];
      py = m.raw_measurements_[1];
      std_px = std_laspx_;
      std_py = std_laspy_;
      break;
    }
    default:
      throw "Unknown sensor type";
  }
  // Initial state vector and process covariance matrix.
  // We are only certain about location (px, py) initially.
  // We don't know anything about velocity and its direction.
  // For turn rate, we assume it does not change initially.
  x_ << px, py, 0, 0, 0;
  P_ << std_px, 0, 0, 0, 0,
        0, std_py, 0, 0, 0,
        0, 0, 10, 0, 0,
        0, 0, 0, 10, 0,
        0, 0, 0, 0, 1;

  // Record the first timestamp.
  previous_timestamp_ = m.timestamp_;
}

double UKF::ProcessMeasurement(const MeasurementPackage& m) {
  if (!is_initialized_) {
    cout << "UKF initialization" << endl;
    InitializeMeasurement(m);
    is_initialized_ = true;
    return -1;
  }
  
  bool isRadar;
  switch (m.sensor_type_) {
    case MeasurementPackage::RADAR: {
      if (!use_radar_) {
        cout << "Skipping radar measurement"<<endl;
        return -1;
      }
      isRadar = true;
      break;
    }
    case MeasurementPackage::LASER: {
      if (!use_laser_) {
        cout << "Skipping laser measurement"<<endl;
        return -1;
      }
      isRadar = false;
      break;
    }
    default:
      throw "Unknown sensor type";
  }

  const double dt = (m.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = m.timestamp_;

  // Prediction
  MatrixXd Xsig_aug;
  Utils::GenerateSigmaPoints(n_aug_, x_, P_, lambda_, std_a_, std_yawdd_, /*output*/ Xsig_aug);

  MatrixXd Xsig_pred;
  Utils::PredictSigmaPoints(n_x_, Xsig_aug, dt, /*output*/ Xsig_pred);

  Utils::GetPredictionMeanAndCovariance(weights_, Xsig_pred, /*output*/ x_, P_);

  // Update
  VectorXd std_noise;
  if (isRadar) {
    std_noise = VectorXd(3);
    std_noise << std_radr_, std_radphi_, std_radrd_;
  } else {
    std_noise = VectorXd(2);
    std_noise << std_laspx_, std_laspy_;
  }
  VectorXd z_pred;
  MatrixXd Zsig, S;
  Utils::GetMeasurementMeanAndCovariance(isRadar, weights_, Xsig_pred, std_noise,
                                         /*output*/ z_pred, Zsig, S);

  double nis;
  Utils::UpdateStates(isRadar, weights_, Xsig_pred, Zsig, z_pred, m.raw_measurements_, S,
                      /*inout*/ x_, P_, nis);

  cout << "x=\n" << x_ << endl;
  cout << "P=\n" << P_ << endl;
  return nis;
}