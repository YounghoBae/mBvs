#ifndef __SIGMASQ_H__
#define __SIGMASQ_H__

#include <Rcpp.h>
#include <RcppEigen.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

//[[Rcpp::depends(RcppEigen)]]

// Obtaining namespace of MCMCpack package
Rcpp::Environment MCMCpack = Rcpp::Environment::namespace_env("MCMCpack");

// Picking up rinvgamma() and rdirichlet() function from MCMCpack package
Rcpp::Function rinvgamma  = MCMCpack["rinvgamma"];

// Library Functions
double update_sigmaSq(const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& delta, const VectorXd& omega, const MatrixXd& X, const VectorXd& alpha, const double alpha_p, const VectorXd& treat, const VectorXd& psi, const double nu1, const double sigmaSq1);

double update_sigmaSq_Sigma(const MatrixXd& M, const MatrixXd& beta0, const VectorXd& treat, const VectorXd& tau, const VectorXd& gamma, const MatrixXd& X, const MatrixXd& B, const MatrixXd& Sigma_lambda_inv, const VectorXd& nu, const double nu0, const double h_lambda, const VectorXd& lambda, const double mu_lambda, const double sigmaSq0);

// Functions Define
// 1. sigmaSq update
double update_sigmaSq(const VectorXd& y,
                      const VectorXd& alpha0,
                      const MatrixXd& M,
                      const VectorXd& delta,
                      const VectorXd& omega,
                      const MatrixXd& X,
                      const VectorXd& alpha,
                      const double alpha_p,
                      const VectorXd& treat,
                      const VectorXd& psi,
                      const double nu1,
                      const double sigmaSq1) {
  int n = M.rows();
  // double idx;
  VectorXd res_y = y - alpha0 - M * delta - X * alpha - alpha_p * treat;
  double y_part  = res_y.squaredNorm();

  double update_shape = (n + omega.sum() + nu1)/2.0;
  
  VectorXd update_scale_part1 = omega.array() * (delta.array() / psi.array());
  double update_scale_part2 = update_scale_part1.squaredNorm();
  double update_scale      = (y_part + update_scale_part2 + nu1 * sigmaSq1)/2.0;
  
  Rcpp::NumericVector z = rinvgamma(n = 1, update_shape, update_scale);
  double z1 = z(0);
  return z1;
}

// 2. sigmaSq_Sigma update
double update_sigmaSq_Sigma(const MatrixXd& M,
                            const MatrixXd& beta0,
                            const VectorXd& treat,
                            const VectorXd& tau,
                            const VectorXd& gamma,
                            const MatrixXd& X,
                            const MatrixXd& B,
                            const MatrixXd& Sigma_lambda_inv,
                            const VectorXd& nu,
                            const double nu0,
                            const double h_lambda,
                            const VectorXd& lambda,
                            const double mu_lambda,
                            const double sigmaSq0){
  int n = M.rows(), q = M.cols();
  
  MatrixXd res_m = M - beta0 - (treat * tau.transpose()) - (X * B);
  VectorXd m_part1 = (res_m * Sigma_lambda_inv * res_m.transpose()).diagonal();
  double m_part = m_part1.sum();
  
  double update_shape = (n * q + gamma.sum() + q + nu0)/2.0;

  VectorXd update_scale_part = gamma.array() * (tau.array().square() / (nu.array().square() * (lambda.array().square() + 1.0)));

  VectorXd mu_lambda_vec = VectorXd::Constant(q, mu_lambda);
  VectorXd res_lambda = lambda - mu_lambda_vec;
  double lambda_part = res_lambda.squaredNorm();

  double update_scale = (m_part + update_scale_part.sum() + lambda_part / h_lambda + (nu0 * sigmaSq0))/2.0;

  Rcpp::NumericVector z = rinvgamma(1, update_shape, update_scale);
  double z1 = z(0);
  return z1;
}


#endif