#pragma once

#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Utils {
public:
  /**
  * Constructor.
  */
  Utils();

  /**
  * Destructor.
  */
  virtual ~Utils();

  /**
  * A helper method to calculate RMSE.
  */
  static VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                                const vector<VectorXd> &ground_truth);

};