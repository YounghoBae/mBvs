//basic.h
#ifndef __BASIC_H__
#define __BASIC_H__

#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Dense>
#include <random>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// [[Rcpp::depends(RcppEigen)]]

// Library Functions
MatrixXd sampleMultivariateNormal(const VectorXd& mean, const MatrixXd& cov, int n_samples);

double normal_dist(const double mu, const double sigma2);
double log_normal_dist(const double mu, const double sigma2);

VectorXd zero_index(const VectorXd& x);
int random_zero_index(const VectorXd& x);

VectorXd nonzero_index(const VectorXd& x);
int random_nonzero_index(const VectorXd& x);

int ran_sample(const int x);
VectorXd ran_sample_vec(const int x);

// Functions Define
// 1. Sampling from multivariate normal dist.
MatrixXd sampleMultivariateNormal(const VectorXd& mean, const MatrixXd& cov, int n_samples) {
  int n = mean.size();
  Eigen::LLT<MatrixXd> cholesky(cov);
  
  if (cholesky.info() == Eigen::Success) {
    MatrixXd Z = MatrixXd::Random(n, n_samples);
    return (mean.replicate(1, n_samples) + cholesky.matrixL() * Z).transpose();
  } else {
    std::cerr << "Cholesky decomposition failed!" << std::endl;
    return Eigen::MatrixXd();
  }
}

// 2. normal_dist
double normal_dist(const double mu, const double sigma2){
    double value = exp(-pow(mu,2) / (2 * sigma2)) / sqrt(M_2PI * sigma2);
    return value;
}

// 3. log_normal_dist
double log_normal_dist(const double mu, const double sigma2){
    double value = -pow(mu,2) / (2 * sigma2) - 0.5 * log(M_2PI * sigma2);
    return value;
}

// 4. random index in zero element
VectorXd zero_index(const VectorXd& x) {
    int z = (x.array() == 0).count(); // Count the number of zero elements
    VectorXd result(z); // Create a vector to store indices

    int k = 0; // Index for storing indices in result
    for (int j = 0; j < x.rows(); j++) {
        if (x(j) == 0) {
            result(k) = j; // Store index of zero element
            k++;
        }
    }
    return result; 
}

// 5. random index in zero element
int random_zero_index(const VectorXd& x) {
    int z = (x.array() == 0).count(); // Count the number of zero elements
    VectorXd result(z); // Create a vector to store indices

    int k = 0; // Index for storing indices in result
    for (int j = 0; j < x.rows(); j++) {
        if (x(j) == 0) {
            result(k) = j; // Store index of zero element
            k++;
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, z - 1);
    return result(dis(gen)); 
}

// 6. random index in nonzero element
VectorXd nonzero_index(const VectorXd& x) {
    int z = (x.array() != 0).count(); // Count the number of zero elements
    VectorXd result(z); // Create a vector to store indices

    int k = 0; // Index for storing indices in result
    for (int j = 0; j < x.rows(); j++) {
        if (x(j) != 0) {
            result(k) = j; // Store index of zero element
            k++;
        }
    }
    return result; 
}

// 7. random index in nonzero element
int random_nonzero_index(const VectorXd& x) {
    int z = (x.array() != 0).count(); // Count the number of zero elements
    VectorXd result(z); // Create a vector to store indices

    int k = 0; // Index for storing indices in result
    for (int j = 0; j < x.rows(); j++) {
        if (x(j) != 0) {
            result(k) = j; // Store index of zero element
            k++;
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, z - 1);
    return result(dis(gen)); 
}

// 7-2. random index in zero element given vector gamma
int random_zero_index_given(const VectorXd& x, const VectorXd& gamma) {
  int max_num = (gamma.array() == 1).count();
  VectorXd gam_ome = gamma.cwiseProduct(x);
  int omega_one_num = (gam_ome.array() == 1).count();
  int can_idx_num = max_num - omega_one_num;
  
  VectorXd result(can_idx_num);
  
  int k=0;
  for(int j=0; j<x.rows(); j++){
    if(x(j) == 0 && gamma(j) == 1){
      result(k) = j;
      k++;
    }
  }
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, can_idx_num - 1);
  return result(dis(gen)); 
}

// 7-3. random index in zero element given vector gamma
int random_nonzero_index_given(const VectorXd& x, const VectorXd& gamma) {

  VectorXd gam_ome = gamma.cwiseProduct(x);
  int omega_one_num = (gam_ome.array() == 1).count();
  
  VectorXd result(omega_one_num);
  
  int k=0;
  for(int j=0; j<x.rows(); j++){
    if(x(j) == 1 && gamma(j) == 1){
      result(k) = j;
      k++;
    }
  }
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, omega_one_num - 1);
  return result(dis(gen)); 
}


// 8. random sample from c(1,2,...,x)
int ran_sample(const int x){
  int random_index = Eigen::internal::random<int>(1, x);
  return random_index;
}

// 9. random samples from c(1,2,...,x)
VectorXd ran_sample_vec(const int x){
  VectorXd vector1To300 = VectorXd::LinSpaced(x, 1, x);
  
  int numSamples = 5;
  
  std::random_device rd;
  std::mt19937 gen(rd());
  
  std::vector<int> indices(vector1To300.size());
  std::iota(indices.begin(), indices.end(), 0);
  
  // Shuffle the indices
  std::shuffle(indices.begin(), indices.end(), gen);
  
  // Create a vector to store the sampled numbers
  Eigen::VectorXd sampledNumbers(numSamples);
  
  // Sample random numbers from the vector
  for (int i = 0; i < numSamples; ++i) {
    sampledNumbers(i) = vector1To300(indices[i]) - 1;
  }
  
  return sampledNumbers; 
}

#endif