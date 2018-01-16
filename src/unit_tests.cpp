#include <cassert>
#include <iostream>
#include <cmath>
#include "Eigen/Dense"
#include "utils.h"

using namespace std;

void AssertVectorEquals(const VectorXd& actual, const VectorXd& expect, int row=0) {
  assert(actual.size() == expect.size());
  for (int j = 0; j < actual.cols(); j++) {
    if (fabs(actual(j) - expect(j)) >= 0.0001) {
      char buf[128];
      sprintf(buf, "[%d, %d] actual=%f expect=%f", row, j, actual(j), expect(j));
      cerr<<buf<<endl;
      assert(false);
    }
  }
}

void AssertMatrixEquals(const MatrixXd& actual, const MatrixXd& expect) {
  assert(actual.rows() == expect.rows() && actual.cols() == expect.cols());
  for (int i = 0; i < actual.rows(); i++) {
    AssertVectorEquals(actual.row(i), expect.row(i), i);
  }
}

void TestGenerateSigmaPoints() {
  // Set state dimension
  int n_x = 5;
  // Set augmented dimension
  int n_aug = 7;
  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a = 0.2;
  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd = 0.2;
  // Define spreading parameter
  double lambda = 3 - n_aug;
  // Set example state
  VectorXd x = VectorXd(n_x);
  x <<   5.7441,
         1.3800,
         2.2049,
         0.5015,
         0.3528;
  // Create example covariance matrix
  MatrixXd P = MatrixXd(n_x, n_x);
  P <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

  MatrixXd Xsig_aug;
  Utils::GenerateSigmaPoints(n_aug, x, P, lambda, std_a, std_yawdd, Xsig_aug);

  // Expected result
  MatrixXd expected(n_aug, n_aug * 2 + 1);
  expected <<
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,  5.7441,  5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,  1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,  2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,  0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641;

  AssertMatrixEquals(Xsig_aug, expected);
}

void TestPredictSigmaPoints() {
  // Set state dimension
  int n_x = 5;
  // Set augmented dimension
  int n_aug = 7;
  // Create example sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
  Xsig_aug <<
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;
  // Time diff in sec
  double dt = 0.1;

  MatrixXd Xsig_pred;
  Utils::PredictSigmaPoints(n_x, Xsig_aug, dt, Xsig_pred);

  // Expected value
  MatrixXd expected = MatrixXd(n_x, 2 * n_aug + 1);
  expected <<
    5.93553, 6.06251, 5.92217, 5.9415, 5.92361, 5.93516, 5.93705, 5.93553, 5.80832, 5.94481, 5.92935, 5.94553, 5.93589, 5.93401, 5.93553,
    1.48939, 1.44673, 1.66484, 1.49719, 1.508, 1.49001, 1.49022, 1.48939, 1.5308, 1.31287, 1.48182, 1.46967, 1.48876, 1.48855, 1.48939,
    2.2049, 2.28414, 2.24557, 2.29582, 2.2049, 2.2049, 2.23954, 2.2049, 2.12566, 2.16423, 2.11398, 2.2049, 2.2049, 2.17026, 2.2049,
    0.53678, 0.473387, 0.678098, 0.554557, 0.643644, 0.543372, 0.53678, 0.538512, 0.600173, 0.395462, 0.519003, 0.429916, 0.530188, 0.53678, 0.535048,
    0.3528, 0.299973, 0.462123, 0.376339, 0.48417, 0.418721, 0.3528, 0.387441, 0.405627, 0.243477, 0.329261, 0.22143, 0.286879, 0.3528, 0.318159;
    
  AssertMatrixEquals(Xsig_pred, expected);
}

void TestPredictMeanAndCovariance() {
  // Set state dimension
  int n_x = 5;
  // Set augmented dimension
  int n_aug = 7;
  // Define spreading parameter
  double lambda = 3 - n_aug;
  // Create example matrix with predicted sigma points
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
  Xsig_pred <<
    5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
    1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
    2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
    0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
    0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

  // Create vector for weights
  VectorXd weights;
  Utils::GetWeights(n_aug, lambda, weights);

  VectorXd x;
  MatrixXd P;
  Utils::PredictMeanAndCovariance(weights, Xsig_pred, x, P);

  // Expected results
  VectorXd expected_x(n_x);
  expected_x << 5.93637, 1.49035, 2.20528, 0.536853, 0.353577;
  MatrixXd expected_P(n_x, n_x);
  expected_P <<
    0.00543425, -0.0024053, 0.00341576, -0.00348196, -0.00299378,
    -0.0024053, 0.010845, 0.0014923, 0.00980182, 0.00791091,
    0.00341576, 0.0014923, 0.00580129, 0.000778632, 0.000792973,
    -0.00348196, 0.00980182, 0.000778632, 0.0119238, 0.0112491,
   -0.00299378, 0.00791091, 0.000792973, 0.0112491, 0.0126972;

  AssertVectorEquals(x, expected_x);
  AssertMatrixEquals(P, expected_P);
}

int main() {
  TestGenerateSigmaPoints();
  TestPredictSigmaPoints();
  TestPredictMeanAndCovariance();

  cout<<"All tests passed"<<endl;
  return 0;
}