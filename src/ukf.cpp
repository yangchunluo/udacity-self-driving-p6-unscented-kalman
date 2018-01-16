#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
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

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // FIXME: to tune
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // FIXME: to tune
  std_yawdd_ = 30;
  
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
  switch (m.sensor_type_) {
    case MeasurementPackage::RADAR: {
      // Convert radar from polar to cartesian coordinates. Ignore velocity informaiton.
      px = m.raw_measurements_[0] * sin(m.raw_measurements_[1]);
      py = m.raw_measurements_[0] * cos(m.raw_measurements_[1]);
      break;
    }
    case MeasurementPackage::LASER: {
      // Laser has position (px, py) but no velocity information.
      px = m.raw_measurements_[0];
      py = m.raw_measurements_[1];
      break;
    }
    default:
      throw "Unknown sensor type";
  }
  // Initial state vector.
  x_ << px, py, 0, 0, 0;

  // Initialize process covariance matrix. We are only certain about location (px, py) at this point.
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1000, 0, 0,
        0, 0, 0, 1000, 0,
        0, 0, 0, 0, 1000;

  // Record the first timestamp.
  previous_timestamp_ = m.timestamp_;
}

void UKF::ProcessMeasurement(const MeasurementPackage& m) {
  if (!is_initialized_) {
    cout << "UKF initialization" << endl;
    InitializeMeasurement(m);
    is_initialized_ = true;
    return;
  }
  
  if (!use_laser_ && m.sensor_type_ == MeasurementPackage::LASER) {
    cout << "Skipping laser measurement";
    return;
  }
  if (!use_radar_ && m.sensor_type_ == MeasurementPackage::RADAR) {
    return;
  }

  double dt = m.timestamp_ - previous_timestamp_;
  previous_timestamp_ = m.timestamp_;

  Prediction(dt);

  switch (m.sensor_type_) {
    case MeasurementPackage::RADAR: {
      UpdateRadar(m);
      break;
    }
    case MeasurementPackage::LASER: {
      UpdateLidar(m);
      break;
    }
    default:
      throw "Unknown sensor type";
  }
}

static inline double squared(double x) {
  return x * x;
}

void UKF::Prediction(double dt) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  /******************************************************************************
   * 1. Generate sigma points
   ******************************************************************************/
  const int naug = n_aug_ - n_x_;

  // Augmented mean state vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug.tail(naug) << 0, 0;

  // Augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(naug, naug) << squared(std_a_), 0,
                                         0, squared(std_yawdd_);

  // Square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  // Augmented sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.col(0) = x_aug;
  double scaler = sqrt(lambda_ + n_aug_);
  for (int i = 0; i < n_aug_; i++) {
      Xsig_aug.col(1 + i) = x_aug + scaler * A.col(i);
      Xsig_aug.col(1 + n_aug_ + i) = x_aug - scaler * A.col(i);
  }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const MeasurementPackage& m) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage& m) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
