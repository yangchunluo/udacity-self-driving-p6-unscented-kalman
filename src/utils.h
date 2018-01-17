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

  /** A standalone method to get prediction mean and covariance. */
  static void GetPredictionMeanAndCovariance(const VectorXd& weights, const MatrixXd Xsig_pred,
                                             VectorXd& out_x, MatrixXd &out_P);

  /** A standalone method to get radar measurement mean and covariance. */
  static void GetRadarMeasurementMeanAndCovariance(const VectorXd& weights, const MatrixXd Xsig_pred,
                                                   const VectorXd& std_radar_noise,
                                                   VectorXd& out_zpred,  MatrixXd& out_Zsig,
                                                   MatrixXd& out_S);

  /** A standalone method to update states. */
  static void UpdateStates(const VectorXd& weights, const MatrixXd& Xsig_pred, const MatrixXd& Zsig,
                           const VectorXd& z_pred, const VectorXd& z, const MatrixXd& S,
                           VectorXd& x, MatrixXd& P);
};