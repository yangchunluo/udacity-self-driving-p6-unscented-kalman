#include <iostream>
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