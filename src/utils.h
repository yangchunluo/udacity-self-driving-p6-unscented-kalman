#pragma once

#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Utils {
public:
  /** Constructor. */
  Utils();

  /** Destructor. */
  virtual ~Utils();

  /** A helper method to calculate RMSE. */
  static VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                                const vector<VectorXd> &ground_truth);

  /** A standalone method to generate sigma points. */
  static void GenerateSigmaPoints(int n_aug, const VectorXd& x, const MatrixXd& P,
                                  int lambda, double std_a, double std_yawdd, MatrixXd& out);

  /** A standalone method to predict sigma points. */
  static void PredictSigmaPoints(int n_x, const MatrixXd& Xsig_aug, double dt, MatrixXd& out);

  /** A standalone methold to get the weight vector. */
  static void GetWeights(int n_aug, int lambda, VectorXd& out);

  /** A standalone method to predict mean and covariance. */
  static void PredictMeanAndCovariance(const VectorXd& weights, const MatrixXd Xsig_pred,
                                       VectorXd& out_x, MatrixXd &out_P);
};