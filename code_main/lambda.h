#ifndef __LAMBDA_H__
#define __LAMBDA_H__

#include <Rcpp.h>
#include <RcppEigen.h>
#include "basic.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// [[Rcpp::depends(RcppEigen)]]

// Library Functions
double U_function(const VectorXd& lambda, const double sigmaSq_Sigma, const MatrixXd& M, const MatrixXd& beta0, const VectorXd& tau, const VectorXd& gamma, const VectorXd& treat, const MatrixXd& B, const MatrixXd& X, const double h_lambda, const double mu_lambda, const VectorXd& nu, const double theta_gamma, const double eta);
VectorXd lambda_update(const VectorXd& lambda, const double sigmaSq_Sigma, const MatrixXd& M, const MatrixXd& beta0, const VectorXd& tau, const VectorXd& gamma, const VectorXd& treat, const MatrixXd& B, const MatrixXd& X, const double h_lambda, const double mu_lambda, const VectorXd& nu, const double theta_gamma, const double eta, const double V_lambda);
VectorXd lambda_update2(const VectorXd& lambda, const double sigmaSq_Sigma, const MatrixXd& M, const MatrixXd& beta0, const VectorXd& tau, const VectorXd& gamma, const VectorXd& treat, const MatrixXd& B, const MatrixXd& X, const double h_lambda, const double mu_lambda, const VectorXd& nu, const double theta_gamma, const double eta, const double V_lambda);

// Functions Define
// 1. U_function
double U_function(const VectorXd& lambda,
                  const double sigmaSq_Sigma,
                  const MatrixXd& M,
                  const MatrixXd& beta0,
                  const VectorXd& tau,
                  const VectorXd& gamma,
                  const VectorXd& treat,
                  const MatrixXd& B,
                  const MatrixXd& X,
                  const double h_lambda,
                  const double mu_lambda,
                  const VectorXd& nu,
                  const double theta_gamma,
                  const double eta){
  int n = M.rows(), q = lambda.rows();
  
  double Sigma_lambda_det    = lambda.squaredNorm() + 1.0;
  MatrixXd Sigma_lambda_part = lambda * lambda.transpose();
  MatrixXd Sigma_lambda_inv  = MatrixXd::Identity(q,q) - (Sigma_lambda_part / Sigma_lambda_det);
  MatrixXd res_m             = M - beta0 - (treat * tau.transpose()) - (X * B);
  
  double part_1 = (n/2.0) * log(Sigma_lambda_det);
  
  VectorXd part_2_sub = (res_m * Sigma_lambda_inv * res_m.transpose()).diagonal();
  double part_2 = part_2_sub.sum() / (2.0 * sigmaSq_Sigma);
  
  VectorXd mu_lambda_vec = VectorXd::Constant(q, mu_lambda);
  VectorXd res_lambda    = lambda - mu_lambda_vec;
  double part_3_sub      = res_lambda.squaredNorm();
  double part_3          = part_3_sub / (2.0 * h_lambda * sigmaSq_Sigma);
  
  VectorXd part_4_sub1 = lambda.array().square() + 1.0;
  VectorXd part_4_sub2 = part_4_sub1.array().log() * gamma.array();
  double part_4 = part_4_sub2.sum() / 2.0;

  VectorXd part_5_sub = gamma.array() * (tau.array().square() / (nu.array().square() * (lambda.array().square() + 1.0) * sigmaSq_Sigma));
  double part_5 = part_5_sub.sum() / 2.0;
  
  // ---------------------------------------------------------------------------
  MatrixXd Sigma_lambda  = lambda * lambda.transpose() + MatrixXd::Identity(q, q);
  MatrixXd Sigma_cor_sub = MatrixXd::Identity(q,q);
  for(int r=0; r<q; r++){Sigma_cor_sub(r,r) = 1.0 / sqrt(pow(lambda(r),2) + 1.0);}
  MatrixXd Sigma_cor = Sigma_cor_sub * Sigma_lambda.cwiseAbs() * Sigma_cor_sub;
  MatrixXd C_mat = Sigma_cor -  MatrixXd::Identity(q, q);
  
  VectorXd theta_gamma_vec = VectorXd::Constant(q, theta_gamma);
  VectorXd alpha1 = (C_mat * gamma) * eta + theta_gamma_vec;
  VectorXd alpha2_sub = alpha1.array().exp() + 1.0;
  VectorXd alpha2 = alpha2_sub.array().log();

  double part_6 = alpha2.sum() - alpha1.dot(gamma);
  
  double log_post = part_1 + part_2 + part_3 + part_4 + part_5 + part_6;
  return(log_post);
}

// 2. lambda update
VectorXd lambda_update(const VectorXd& lambda,
                       const double sigmaSq_Sigma,
                       const MatrixXd& M,
                       const MatrixXd& beta0,
                       const VectorXd& tau,
                       const VectorXd& gamma,
                       const VectorXd& treat,
                       const MatrixXd& B,
                       const MatrixXd& X,
                       const double h_lambda,
                       const double mu_lambda,
                       const VectorXd& nu,
                       const double theta_gamma,
                       const double eta,
                       const double V_lambda){
  int q = lambda.rows();
  VectorXd out1 = VectorXd::Zero(q);
  
  int update_idx = ran_sample(q) - 1;
  VectorXd new_lambda = lambda;
  new_lambda(update_idx) = Rcpp::rnorm(1, lambda(update_idx), sqrt(V_lambda))(0);
  
  double update_prob1 = U_function(lambda, sigmaSq_Sigma, M, beta0, tau, gamma, treat, B, X, h_lambda, mu_lambda, nu, theta_gamma, eta);
  double update_prob2 = U_function(new_lambda, sigmaSq_Sigma, M, beta0, tau, gamma, treat, B, X, h_lambda, mu_lambda, nu, theta_gamma, eta);

  double update_prob = update_prob1 - update_prob2;
  double log_u = log(Rcpp::runif(1)(0));

  if(update_prob > log_u){
    out1 = new_lambda;
  }else{
    out1 = lambda;
  }
  
  return out1;
}

// lambda update with sample vector together
// 2. lambda update
VectorXd lambda_update2(const VectorXd& lambda,
                       const double sigmaSq_Sigma,
                       const MatrixXd& M,
                       const MatrixXd& beta0,
                       const VectorXd& tau,
                       const VectorXd& gamma,
                       const VectorXd& treat,
                       const MatrixXd& B,
                       const MatrixXd& X,
                       const double h_lambda,
                       const double mu_lambda,
                       const VectorXd& nu,
                       const double theta_gamma,
                       const double eta,
                       const double V_lambda){
  int q = lambda.rows(), idx;
  VectorXd out1 = VectorXd::Zero(q);
  
  VectorXd update_idx = ran_sample_vec(q);

  VectorXd new_lambda = lambda;
  for(int i=0; i<update_idx.rows(); i++){
    idx = update_idx(i);
    new_lambda(idx) = Rcpp::rnorm(1, lambda(idx), sqrt(V_lambda))(0);
  }

  double update_prob1 = U_function(lambda, sigmaSq_Sigma, M, beta0, tau, gamma, treat, B, X, h_lambda, mu_lambda, nu, theta_gamma, eta);
  double update_prob2 = U_function(new_lambda, sigmaSq_Sigma, M, beta0, tau, gamma, treat, B, X, h_lambda, mu_lambda, nu, theta_gamma, eta);
  
  double update_prob = update_prob1 - update_prob2;
  double log_u = log(Rcpp::runif(1)(0));
  
  if(update_prob > log_u){
    out1 = new_lambda;
  }else{
    out1 = lambda;
  }
  
  return out1;
}

#endif