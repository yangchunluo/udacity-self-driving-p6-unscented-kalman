#include <cassert>
#include <iostream>
#include <cmath>
#include "Eigen/Dense"
#include "utils.h"

using namespace std;

void AssertMatrixEquals(const MatrixXd& actual, const MatrixXd& expect) {
  assert(actual.rows() == expect.rows() && actual.cols() == expect.cols());
  for (int i = 0; i < actual.rows(); i++) {
    for (int j = 0; j < actual.cols(); j++) {
        assert(fabs(actual(i, j) - expect(i, j)) < 0.0001);
    }
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

int main() {
  TestGenerateSigmaPoints();

  cout<<"All test passed"<<endl;
  return 0;
}