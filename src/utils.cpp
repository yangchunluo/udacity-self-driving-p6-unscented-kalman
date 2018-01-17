#include <iostream>
#include "Eigen/Dense"
#include "utils.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Utils::Utils() {}

Utils::~Utils() {}

VectorXd Utils::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  if (estimations.size() == 0) {
      throw "the estimation vector size should not be zero";
  }
  if (estimations.size() != ground_truth.size()) {
      throw "the estimation vector size should equal ground truth vector size";
  }

  // Initialization
  VectorXd rmse(ground_truth[0].size());
  for (int i = 0; i < rmse.size(); i++) {
    rmse(i) = 0;
  }

  // Accumulate squared residuals
  for(int i = 0; i < estimations.size(); i++) {
    VectorXd diff = ground_truth[i] - estimations[i].head(rmse.size());
    diff = diff.array() * diff.array();
    rmse += diff;
  }

  // Calculate the mean
  rmse /= estimations.size();

  // Calculate the squared root
  return rmse.array().sqrt();
}

static inline double squared(double x) {
  return x * x;
}

void Utils::GenerateSigmaPoints(int n_aug, const VectorXd& x, const MatrixXd& P,
                                int lambda, double std_a, double std_yawdd,
                                MatrixXd& out) {
  const int n_x = x.size();
  const int naug = n_aug - n_x;

  // Augmented mean state vector
  VectorXd x_aug = VectorXd(n_aug);
  x_aug.head(n_x) = x;
  x_aug.tail(naug) << 0, 0;

  // Augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug, n_aug);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x, n_x) = P;
  P_aug.bottomRightCorner(naug, naug) << squared(std_a), 0,
                                         0, squared(std_yawdd);

  // Square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  // Augmented sigma point matrix (output)
  out = MatrixXd(n_aug, 2 * n_aug + 1);
  out.col(0) = x_aug;
  double scaler = sqrt(lambda + n_aug);
  for (int i = 0; i < n_aug; i++) {
      out.col(1 + i) = x_aug + scaler * A.col(i);
      out.col(1 + n_aug + i) = x_aug - scaler * A.col(i);
  }
}

void Utils::PredictSigmaPoints(int n_x, const MatrixXd& Xsig_aug, double dt,
                               MatrixXd& out) {
  out = MatrixXd(n_x, Xsig_aug.cols());
  for (int i = 0; i < Xsig_aug.cols(); i++) {
    double px = Xsig_aug(0, i),
           py = Xsig_aug(1, i),
            v = Xsig_aug(2, i),
          yaw = Xsig_aug(3, i),
         yawd = Xsig_aug(4, i),
           va = Xsig_aug(5, i),
        yawdd = Xsig_aug(6, i);
      
    // Process function
    VectorXd pf(n_x);
    pf(0) = fabs(yawd < 0.000001) ? v * cos(yaw) * dt : v / yawd * (sin(yaw + yawd * dt) - sin(yaw));
    pf(1) = fabs(yawd < 0.000001) ? v * sin(yaw) * dt : v / yawd * (-cos(yaw + yawd * dt) + cos(yaw));
    pf(2) = 0;
    pf(3) = yawd * dt;
    pf(4) = 0;
      
    // Process noise
    VectorXd pn(n_x);
    double dt2 = squared(dt);
    pn(0) = 0.5 * dt2 * cos(yaw) * va;
    pn(1) = 0.5 * dt2 * sin(yaw) * va;
    pn(2) = dt * va;
    pn(3) = 0.5 * dt2 * yawdd;
    pn(4) = dt * yawdd;
      
    out.col(i) = Xsig_aug.col(i).head(n_x) + pf + pn;
  }
}

void Utils::GetWeights(int n_aug, int lambda, VectorXd& out) {
  out = VectorXd(2 * n_aug + 1);
  out(0) = ((double)lambda) / (lambda + n_aug);
  for (int i = 1; i < out.size(); i++) {
    out(i) = 0.5 / (lambda + n_aug);
  }
}

static inline void NormalizeAngle(double& angle) {
  const double PI = 3.14159265358979;
  while (angle > PI) angle -= 2 * PI;
  while (angle < -PI) angle += 2 * PI;
}

void Utils::GetPredictionMeanAndCovariance(const VectorXd& weights, const MatrixXd Xsig_pred,
                                           VectorXd& out_x, MatrixXd &out_P) {
  const int n_x = Xsig_pred.rows();
  
  // Predict state mean
  out_x = Xsig_pred * weights;

  // Predict state covariance matrix
  out_P = MatrixXd(n_x, n_x);
  out_P.fill(0.0);
  for (int i = 0; i < weights.size(); i++) {
    // State difference
    VectorXd x_diff = Xsig_pred.col(i) - out_x;
    // Angle normalization
    NormalizeAngle(x_diff(3));

    out_P += weights(i) * x_diff * x_diff.transpose();
  }
}

void Utils::GetRadarMeasurementMeanAndCovariance(const VectorXd& weights, const MatrixXd Xsig_pred,
                                                 const VectorXd& std_radar_noise,
                                                 VectorXd& out_zpred,  MatrixXd& out_Zsig,
                                                 MatrixXd& out_S) {
  const int n_z = std_radar_noise.size(); 
  out_Zsig = MatrixXd(n_z, Xsig_pred.cols());
  // Transform sigma points into measurement space
  for (int i = 0; i < Xsig_pred.cols(); i++) {
    double px = Xsig_pred(0, i),
           py = Xsig_pred(1, i),
            v = Xsig_pred(2, i),
          yaw = Xsig_pred(3, i),
         yawd = Xsig_pred(4, i);
      
      VectorXd zsig(n_z);
      zsig(0) = sqrt(squared(px) + squared(py));                   // r
      zsig(1) = atan2(py, px);                                     // phi
      zsig(2) = (px * cos(yaw) * v + py * sin(yaw) * v) / zsig(0); // dot_r

      out_Zsig.col(i) = zsig;
  }
  
  // Calculate mean predicted measurement
  out_zpred = out_Zsig * weights;
  
  // Calculate innovation covariance matrix S
  out_S = MatrixXd(n_z, n_z);
  out_S.fill(0);
  for (int i = 0; i < weights.size(); i++) {
    // Measurement residual
    VectorXd z_diff = out_Zsig.col(i) - out_zpred;
    // Angle normozation
    NormalizeAngle(z_diff(1));

    out_S += weights(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  for (int i = 0; i < n_z; i++) {
    out_S(i, i) += squared(std_radar_noise(i));
  }
}

void Utils::GetLaserMeasurementMeanAndCovariance(const VectorXd& weights, const MatrixXd Xsig_pred,
                                                 const VectorXd& std_laser_noise,
                                                 VectorXd& out_zpred,  MatrixXd& out_Zsig,
                                                 MatrixXd& out_S) {
  const int n_z = std_laser_noise.size(); 
  out_Zsig = MatrixXd(n_z, Xsig_pred.cols());
  // Transform sigma points into measurement space
  for (int i = 0; i < Xsig_pred.cols(); i++) {
    double px = Xsig_pred(0, i),
           py = Xsig_pred(1, i);  
    out_Zsig.col(i) << px, py;
  }
  
  // Calculate mean predicted measurement
  out_zpred = out_Zsig * weights;
  
  // Calculate innovation covariance matrix S
  out_S = MatrixXd(n_z, n_z);
  out_S.fill(0);
  for (int i = 0; i < weights.size(); i++) {
    // Measurement residual
    VectorXd z_diff = out_Zsig.col(i) - out_zpred;
    out_S += weights(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  for (int i = 0; i < n_z; i++) {
    out_S(i, i) += squared(std_laser_noise(i));
  }
}

void Utils::UpdateStates(const VectorXd& weights, const MatrixXd& Xsig_pred, const MatrixXd& Zsig,
                         const VectorXd& z_pred, const VectorXd& z, const MatrixXd& S,
                         VectorXd& x, MatrixXd& P) {
  const int n_x = x.size(),
            n_z = z.size();

  // Calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x, n_z);
  Tc.fill(0);
  for (int i = 0; i < Xsig_pred.cols(); i++) {
    // Measurement residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // Angle normalization
    NormalizeAngle(z_diff(1));

    // State difference
    VectorXd x_diff = Xsig_pred.col(i) - x;
    // Angle normalization
    NormalizeAngle(x_diff(3));

    Tc += weights(i) * x_diff * z_diff.transpose();
  }
  
  // Calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();
  
  // Residual concerning the actual measurement
  VectorXd z_diff = z - z_pred;
  // Angle normalization
  NormalizeAngle(z_diff(1));

  // Update state mean and covariance matrix
  x += K * z_diff;
  P -= K * S * K.transpose();
}