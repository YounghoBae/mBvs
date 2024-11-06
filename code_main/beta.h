#ifndef __BETA_H__
#define __BETA_H__

#include <Rcpp.h>
#include <RcppEigen.h>
#include "basic.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// [[Rcpp::depends(RcppEigen)]]

// Library Functions
MatrixXd update_beta0(const MatrixXd& M, const VectorXd& treat, const VectorXd& tau, const MatrixXd& X, const MatrixXd& B, const MatrixXd& Sigma_inv, double h0, double mu0);

MatrixXd update_beta (const MatrixXd& M, const MatrixXd& beta0, const VectorXd& treat, const VectorXd& tau, const MatrixXd& X, const MatrixXd& B, const MatrixXd& Sigma_inv, double c0, double mu1);

// Functions Define
// 1. beta0 update
MatrixXd update_beta0(const MatrixXd& M,
                      const VectorXd& treat,
                      const VectorXd& tau,
                      const MatrixXd& X,
                      const MatrixXd& B,
                      const MatrixXd& Sigma_inv,
                      double h0,
                      double mu0) {
  const int n = M.rows();
  const int q = M.cols();
  
  const VectorXd mu0_vec = VectorXd::Constant(q, mu0);
  
  const MatrixXd update_sigma_part = n * Sigma_inv + (1.0 / h0) * MatrixXd::Identity(q, q);
  const MatrixXd update_sigma = update_sigma_part.inverse();
  
  const MatrixXd res_m = M - (treat * tau.transpose()) - (X * B);
  
  const VectorXd update_mu = update_sigma * ((Sigma_inv * res_m.colwise().sum().transpose()) + (1.0 / h0) * mu0_vec);
  
  // Sample from multivariate normal distribution using Eigen
  Eigen::MatrixXd beta0 = sampleMultivariateNormal(update_mu, update_sigma, n);
  
  return beta0;
}

// 2. beta update
MatrixXd update_beta(const MatrixXd& M,
                     const MatrixXd& beta0,
                     const VectorXd& treat,
                     const VectorXd& tau,
                     const MatrixXd& X,
                     const MatrixXd& B,
                     const MatrixXd& Sigma_inv,
                     double c0,
                     double mu1) {
  int n = M.rows(), p = B.rows(), q = B.cols();
  
  MatrixXd res_m(n, q);
  MatrixXd update_sigma_part(p, p);
  MatrixXd update_sigma(p, p);
  VectorXd mu_vec = VectorXd::Constant(p, mu1);
  VectorXd update_mu_part(p);
  VectorXd update_mu(p);
  
  MatrixXd B_j = B;
  for (int j = 0; j < q; j++) {
    B_j.col(j).setZero(); // Set j-th column of B_j to zero
    
    res_m = M - beta0 - treat * tau.transpose() - X * B_j;
    double j_sigma_inv = Sigma_inv(j, j);
    
    update_sigma_part = (X.transpose() * X) * j_sigma_inv + (MatrixXd::Identity(p, p) / c0);
    update_sigma = update_sigma_part.inverse();
    
    update_mu_part = Sigma_inv.row(j) * res_m.transpose() * X;
    update_mu = update_sigma * (update_mu_part + mu_vec / c0);
    
    B_j.col(j) = sampleMultivariateNormal(update_mu, update_sigma, 1).row(0);
  }
  return B_j;
}


#endif