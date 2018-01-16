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
  VectorXd rmse(estimations[0].size());
  for (int i = 0; i < rmse.size(); i++) {
    rmse(i) = 0;
  }

  // Accumulate squared residuals
  for(int i = 0; i < estimations.size(); i++) {
    VectorXd diff = ground_truth[i] - estimations[i];
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