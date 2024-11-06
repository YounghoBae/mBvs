#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <RcppEigen.h>
#include "basic.h"
#include "lambda.h"
#include "beta.h"
#include "alpha.h"
#include "sigmaSq.h"
#include "tau.h"
#include "delta.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export()]]
Rcpp::List cmaVSm (VectorXd treat, MatrixXd M, MatrixXd X, VectorXd y,
                   int iter, int burn_in, int thin,
                   double theta_gamma,
                   double mu_lambda, double h_lambda,
                   double nu_element,
                   double h0, double c0,
                   double V_tau, double V_lambda,
                   double mu0, double mu1, double nu0, double sigmaSq0,
                   VectorXd update_prop, double eta,
                   double init1, double init2, double init3, double init4, double init5){
  
  std::cout << "Initial Setting Start" << "\n";
  
  ////////////// Initial Setting //////////////
  int n = M.rows(), q = M.cols(), p = X.cols(), a=0;
  int b = (iter - burn_in)*1.0 / thin;
  
  // For initial
  double sigmaSq_Sigma = init1;
  VectorXd lambda = VectorXd::Zero(q);
  for(int i=0; i<30; i++){lambda(i) = init2;}
  for(int j=30; j<q; j++){lambda(j) = init3;}
  double Sigma_lambda_det = lambda.squaredNorm() + 1.0;
  MatrixXd Sigma_lambda_inv = MatrixXd::Identity(q, q) - ((lambda * lambda.transpose()) / Sigma_lambda_det);
  MatrixXd Sigma_lambda  = lambda * lambda.transpose() + MatrixXd::Identity(q, q);
  MatrixXd Sigma = sigmaSq_Sigma * Sigma_lambda;
  MatrixXd Sigma_inv = Sigma_lambda_inv / sigmaSq_Sigma;
  
  MatrixXd Sigma_cor_sub = MatrixXd::Identity(q,q);
  for(int r=0; r<q; r++){Sigma_cor_sub(r,r) = 1.0 / sqrt(pow(lambda(r),2) + 1.0);}
  MatrixXd Sigma_cor = Sigma_cor_sub * Sigma_lambda.cwiseAbs() * Sigma_cor_sub;
  MatrixXd C_mat     = Sigma_cor -  MatrixXd::Identity(q, q);
  
  MatrixXd eff_coeff = MatrixXd::Zero(q, 4);
  VectorXd tau = eff_coeff.col(0);
  VectorXd gamma = eff_coeff.col(1);
  
  VectorXd nu  = VectorXd::Constant(q, nu_element);
  
  MatrixXd beta0 = MatrixXd::Constant(n, q, init4);
  MatrixXd B = MatrixXd::Constant(p, q, init5);
  
  // For sample store
  VectorXd sigmaSq_Sigma_sample = VectorXd::Zero(b);
  MatrixXd lambda_sample = MatrixXd::Zero(q, b);

  MatrixXd tau_sample   = MatrixXd::Zero(q, b);
  MatrixXd gamma_sample = MatrixXd::Zero(q, b);

  MatrixXd beta0_sample = MatrixXd::Zero(q, b);
  arma::cube B_sample(p, q, b);
  
  std::cout << "Initial Setting End" << "\n";
  
  std::cout << "Iteration Start" << "\n";
  ////////////// Iteration //////////////
  for(int t=0; t<iter; t++){
    // update tau, gamma
    if(Rcpp::runif(1)(0) < update_prop(0)){
      eff_coeff = update_tau2(eff_coeff, V_tau, nu, lambda, Sigma, C_mat, theta_gamma, M, beta0, treat, X, B, Sigma_inv, eta);
    }
    tau   = eff_coeff.col(0);
    gamma = eff_coeff.col(1);
    
    // update beta0
    if(Rcpp::runif(1)(0) < update_prop(2)){
      beta0 = update_beta0(M, treat, tau, X, B ,Sigma_inv, h0, mu0);
    }
    
    // update B
    if(Rcpp::runif(1)(0) < update_prop(3)){
      B = update_beta(M, beta0, treat, tau, X, B, Sigma_inv, c0, mu1);
    }
    
    // update lambda
    if(Rcpp::runif(1)(0) < update_prop(7)){
      for(int z=0; z<10; z++){
        lambda = lambda_update(lambda, sigmaSq_Sigma, M, beta0, tau, gamma, treat, B, X, h_lambda, mu_lambda, nu, theta_gamma, eta, V_lambda);
      }
    }
    
    Sigma_lambda_det = lambda.squaredNorm() + 1.0;
    Sigma_lambda_inv = MatrixXd::Identity(q, q) - ((lambda * lambda.transpose()) / Sigma_lambda_det);
    Sigma_lambda  = lambda * lambda.transpose() + MatrixXd::Identity(q, q);
    Sigma = sigmaSq_Sigma * Sigma_lambda;
    Sigma_inv = Sigma_lambda_inv / sigmaSq_Sigma;
    
    Sigma_cor_sub = MatrixXd::Identity(q,q);
    for(int r=0; r<q; r++){Sigma_cor_sub(r,r) = 1.0 / sqrt(pow(lambda(r),2) + 1.0);}
    Sigma_cor = Sigma_cor_sub * Sigma_lambda.cwiseAbs() * Sigma_cor_sub;
    C_mat     = Sigma_cor -  MatrixXd::Identity(q, q);
    
    // update sigmaSq_Sigma
    if(Rcpp::runif(1)(0) < update_prop(8)){
      sigmaSq_Sigma = update_sigmaSq_Sigma(M, beta0, treat, tau, gamma, X, B, Sigma_lambda_inv, nu, nu0, h_lambda, lambda, mu_lambda, sigmaSq0);
    }
    
    if(t >= burn_in && t % thin == 0){
      // store sample
      tau_sample.col(a)         = tau;
      gamma_sample.col(a)       = gamma;
      beta0_sample.col(a)       = beta0.row(0);
      B_sample.slice(a)         = Rcpp::as<arma::mat>(Rcpp::wrap(B));
      lambda_sample.col(a)      = lambda;
      sigmaSq_Sigma_sample(a)   = sigmaSq_Sigma;
      a++;
    }
    if((t+1) % 10000 == 0){std::cout << "Iteration completed " << (t+1) << "/" << iter << "\n";}
  }
  std::cout << "Iteration End" << "\n";
  
  std::cout << "Posterior Value Computation Start" << "\n";
  ////////////// Posterior Value Computation //////////////
  VectorXd post_tau           = VectorXd::Zero(q);
  VectorXd post_gamma         = VectorXd::Zero(q);
  VectorXd post_delta         = VectorXd::Zero(q);
  VectorXd post_omega         = VectorXd::Zero(q);
  VectorXd post_include       = VectorXd::Zero(q);
  VectorXd post_beta0         = VectorXd::Zero(q);
  MatrixXd post_beta          = MatrixXd::Zero(p, q);
  double post_alpha0=0.0, post_alpha_p=0.0, post_sigmaSq_Sigma=0.0, post_sigmaSq=0.0;
  VectorXd post_alpha         = VectorXd::Zero(p);
  VectorXd post_lambda        = VectorXd::Zero(q);
  
  post_tau     = tau_sample.rowwise().mean();
  post_gamma   = gamma_sample.rowwise().mean();
  post_beta0   = beta0_sample.rowwise().mean();
  
  arma::mat post_beta_sum (p, q, arma::fill::zeros);
  for(int f=0; f<b; f++){post_beta_sum = post_beta_sum + B_sample.slice(f);}
  post_beta = Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(Rcpp::wrap(post_beta_sum/b));
  
  post_lambda        = lambda_sample.rowwise().mean();
  post_sigmaSq_Sigma = sigmaSq_Sigma_sample.mean();
  
  std::cout << "Posterior Value Computation End" << "\n";
  
  Rcpp::List ret;
  ret["post_tau"] = post_tau;
  ret["post_gamma"] = post_gamma;
  ret["post_delta"] = post_delta;
  ret["post_omega"] = post_omega;
  ret["post_include"] = post_include;
  ret["post_beta0"] = post_beta0;
  ret["post_beta"] = post_beta;
  ret["post_alpha0"] = post_alpha0;
  ret["post_alpha"] = post_alpha;
  ret["post_alpha_p"] = post_alpha_p;
  ret["post_lambda"] = post_lambda;
  ret["post_sigmaSq_Sigma"] = post_sigmaSq_Sigma;
  ret["post_sigmaSq"] = post_sigmaSq;
  ret["tau_sample"] = tau_sample;
  ret["beta0_sample"] = beta0_sample;
  ret["B_sample"] = B_sample;
  ret["lambda_sample"] = lambda_sample;
  ret["sigmaSq_Sigma_sample"] = sigmaSq_Sigma_sample;
  
  return ret;
}

// [[Rcpp::export()]]
Rcpp::List cmaVS_dep (VectorXd treat, MatrixXd M, MatrixXd X, VectorXd y,
                      int iter, int burn_in, int thin,
                      double theta_gamma, double theta_omega,
                      double mu_lambda, double h_lambda,
                      double nu_element, double psi_element,
                      double h0, double c0, double s0, double t0, double k0,
                      double V_tau, double V_delta, double V_lambda,
                      double mu0, double mu1, double nu0, double sigmaSq0, double nu1, double sigmaSq1,
                      VectorXd update_prop, double init, double eta){
  
  std::cout << "Initial Setting Start" << "\n";
  
  ////////////// Initial Setting //////////////
  int n = M.rows(), q = M.cols(), p = X.cols(), a=0;
  int b = (iter - burn_in)*1.0 / thin;
  
  // For initial
  double sigmaSq_Sigma = init;
  VectorXd lambda = VectorXd::Zero(q);
  double Sigma_lambda_det = lambda.squaredNorm() + 1.0;
  MatrixXd Sigma_lambda_inv = MatrixXd::Identity(q, q) - ((lambda * lambda.transpose()) / Sigma_lambda_det);
  MatrixXd Sigma_lambda  = lambda * lambda.transpose() + MatrixXd::Identity(q, q);
  MatrixXd Sigma = sigmaSq_Sigma * Sigma_lambda;
  MatrixXd Sigma_inv = Sigma_lambda_inv / sigmaSq_Sigma;
  
  MatrixXd Sigma_cor_sub = MatrixXd::Identity(q,q);
  for(int r=0; r<q; r++){Sigma_cor_sub(r,r) = 1.0 / sqrt(pow(lambda(r),2) + 1.0);}
  MatrixXd Sigma_cor = Sigma_cor_sub * Sigma_lambda.cwiseAbs() * Sigma_cor_sub;
  MatrixXd C_mat     = Sigma_cor -  MatrixXd::Identity(q, q);
  
  double sigmaSq = init;
  
  MatrixXd eff_coeff = MatrixXd::Zero(q, 4);
  VectorXd tau   = eff_coeff.col(0);
  VectorXd gamma = eff_coeff.col(1);
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  VectorXd include_vector = VectorXd::Zero(q);
  
  VectorXd nu  = VectorXd::Constant(q, nu_element);
  VectorXd psi = VectorXd::Constant(q, psi_element);
  
  MatrixXd beta0 = MatrixXd::Constant(n, q, init);
  MatrixXd B = MatrixXd::Constant(p, q, init);
  
  VectorXd alpha0 = VectorXd::Constant(n, init);
  VectorXd alpha = VectorXd::Constant(p, init);
  double alpha_p = init;
  
  // For sample store
  VectorXd sigmaSq_Sigma_sample = VectorXd::Zero(b);
  MatrixXd lambda_sample = MatrixXd::Zero(q, b);
  VectorXd sigmaSq_sample = VectorXd::Zero(b);
  MatrixXd tau_sample   = MatrixXd::Zero(q, b);
  MatrixXd gamma_sample = MatrixXd::Zero(q, b);
  MatrixXd delta_sample = MatrixXd::Zero(q, b);
  MatrixXd omega_sample = MatrixXd::Zero(q, b);
  MatrixXd include_sample = MatrixXd::Zero(q, b);
  MatrixXd beta0_sample = MatrixXd::Zero(q, b);
  arma::cube B_sample(p, q, b);
  VectorXd alpha0_sample = VectorXd::Zero(b);
  MatrixXd alpha_sample = MatrixXd::Zero(p, b);
  VectorXd alpha_p_sample = VectorXd::Zero(b);
  
  std::cout << "Initial Setting End" << "\n";
  
  std::cout << "Iteration Start" << "\n";
  ////////////// Iteration //////////////
  for(int t=0; t<iter; t++){
    // update tau, gamma
    if(Rcpp::runif(1)(0) < update_prop(0)){
      eff_coeff = update_tau2(eff_coeff, V_tau, nu, lambda, Sigma, C_mat, theta_gamma, M, beta0, treat, X, B, Sigma_inv, eta);
    }
    tau   = eff_coeff.col(0);
    gamma = eff_coeff.col(1);
    
    // update delta, omega
    if(Rcpp::runif(1)(0) < update_prop(1)){
      eff_coeff = update_delta2(eff_coeff, V_delta, psi, sigmaSq, theta_omega, y, alpha0, M, alpha, alpha_p, treat, X);
    }
    delta = eff_coeff.col(2);
    omega = eff_coeff.col(3);
    
    // update beta0
    if(Rcpp::runif(1)(0) < update_prop(2)){
      beta0 = update_beta0(M, treat, tau, X, B ,Sigma_inv, h0, mu0);
    }
    
    // update B
    if(Rcpp::runif(1)(0) < update_prop(3)){
      B = update_beta(M, beta0, treat, tau, X, B, Sigma_inv, c0, mu1);
    }
    
    // update alpha0
    if(Rcpp::runif(1)(0) < update_prop(4)){
      alpha0 = update_alpha0(y, M, delta, X, alpha, alpha_p, treat, sigmaSq, s0);
    }
    
    // update alpha
    if(Rcpp::runif(1)(0) < update_prop(5)){
      alpha = update_alpha(y, alpha0, M, delta, X, alpha_p, treat, sigmaSq, t0);
    }
    
    // update alpha_p
    if(Rcpp::runif(1)(0) < update_prop(6)){
      alpha_p = update_alpha_p(y, alpha0, M, delta, X, alpha, treat, sigmaSq, k0);
    }
    
    // update lambda
    if(Rcpp::runif(1)(0) < update_prop(7)){
      for(int z=0; z<10; z++){
        lambda = lambda_update(lambda, sigmaSq_Sigma, M, beta0, tau, gamma, treat, B, X, h_lambda, mu_lambda, nu, theta_gamma, eta, V_lambda);
      }
    }
    
    Sigma_lambda_det = lambda.squaredNorm() + 1.0;
    Sigma_lambda_inv = MatrixXd::Identity(q, q) - ((lambda * lambda.transpose()) / Sigma_lambda_det);
    Sigma_lambda  = lambda * lambda.transpose() + MatrixXd::Identity(q, q);
    Sigma = sigmaSq_Sigma * Sigma_lambda;
    Sigma_inv = Sigma_lambda_inv / sigmaSq_Sigma;
    
    Sigma_cor_sub = MatrixXd::Identity(q,q);
    for(int r=0; r<q; r++){Sigma_cor_sub(r,r) = 1.0 / sqrt(pow(lambda(r),2) + 1.0);}
    Sigma_cor = Sigma_cor_sub * Sigma_lambda.cwiseAbs() * Sigma_cor_sub;
    C_mat     = Sigma_cor -  MatrixXd::Identity(q, q);
    
    // update sigmaSq_Sigma
    if(Rcpp::runif(1)(0) < update_prop(8)){
      sigmaSq_Sigma = update_sigmaSq_Sigma(M, beta0, treat, tau, gamma, X, B, Sigma_lambda_inv, nu, nu0, h_lambda, lambda, mu_lambda, sigmaSq0);
    }
    
    // update sigmaSq
    if(Rcpp::runif(1)(0) < update_prop(9)){
      sigmaSq = update_sigmaSq(y, alpha0, M, delta, omega, X, alpha, alpha_p, treat, psi, nu1, sigmaSq1);
    }
    
    // include vector
    include_vector = gamma.array() * omega.array();
    
    if(t >= burn_in && t % thin == 0){
      // store sample
      tau_sample.col(a)         = tau;
      gamma_sample.col(a)       = gamma;
      delta_sample.col(a)       = delta;
      omega_sample.col(a)       = omega;
      beta0_sample.col(a)       = beta0.row(0);
      B_sample.slice(a)         = Rcpp::as<arma::mat>(Rcpp::wrap(B));
      alpha0_sample(a)          = alpha0(0);
      alpha_sample.col(a)       = alpha;
      alpha_p_sample(a)         = alpha_p;
      lambda_sample.col(a)      = lambda;
      sigmaSq_Sigma_sample(a)   = sigmaSq_Sigma;
      sigmaSq_sample(a)         = sigmaSq;
      include_sample.col(a)     = include_vector;
      a++;
    }
    if((t+1) % 10000 == 0){
      std::cout << "Iteration completed " << (t+1) << "/" << iter << "\n";
    }
  }
  std::cout << "Iteration End" << "\n";
  
  std::cout << "Posterior Value Computation Start" << "\n";
  ////////////// Posterior Value Computation //////////////
  VectorXd post_tau           = VectorXd::Zero(q);
  VectorXd post_gamma         = VectorXd::Zero(q);
  VectorXd post_delta         = VectorXd::Zero(q);
  VectorXd post_omega         = VectorXd::Zero(q);
  VectorXd post_include       = VectorXd::Zero(q);
  VectorXd post_beta0         = VectorXd::Zero(q);
  MatrixXd post_beta          = MatrixXd::Zero(p, q);
  double post_alpha0=0.0, post_alpha_p=0.0, post_sigmaSq_Sigma=0.0, post_sigmaSq=0.0;
  VectorXd post_alpha         = VectorXd::Zero(p);
  VectorXd post_lambda        = VectorXd::Zero(q);
  
  post_tau     = tau_sample.rowwise().mean();
  post_gamma   = gamma_sample.rowwise().mean();
  post_delta   = delta_sample.rowwise().mean();
  post_omega   = omega_sample.rowwise().mean();
  post_include = include_sample.rowwise().mean();
  post_beta0   = beta0_sample.rowwise().mean();
  
  arma::mat post_beta_sum (p, q, arma::fill::zeros);
  for(int f=0; f<b; f++){post_beta_sum = post_beta_sum + B_sample.slice(f);}
  post_beta = Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(Rcpp::wrap(post_beta_sum/b));
  
  post_alpha0        = alpha0_sample.mean();
  post_alpha_p       = alpha_p_sample.mean();
  post_alpha         = alpha_sample.rowwise().mean();
  post_lambda        = lambda_sample.rowwise().mean();
  post_sigmaSq_Sigma = sigmaSq_Sigma_sample.mean();
  post_sigmaSq       = sigmaSq_sample.mean();
  
  std::cout << "Posterior Value Computation End" << "\n";
  
  Rcpp::List ret;
  ret["post_tau"] = post_tau;
  ret["post_gamma"] = post_gamma;
  ret["post_delta"] = post_delta;
  ret["post_omega"] = post_omega;
  ret["post_include"] = post_include;
  ret["post_beta0"] = post_beta0;
  ret["post_beta"] = post_beta;
  ret["post_alpha0"] = post_alpha0;
  ret["post_alpha"] = post_alpha;
  ret["post_alpha_p"] = post_alpha_p;
  ret["post_lambda"] = post_lambda;
  ret["post_sigmaSq_Sigma"] = post_sigmaSq_Sigma;
  ret["post_sigmaSq"] = post_sigmaSq;
  ret["tau_sample"] = tau_sample;
  ret["delta_sample"] = delta_sample;
  ret["beta0_sample"] = beta0_sample;
  ret["B_sample"] = B_sample;
  ret["alpha0_sample"] = alpha0_sample;
  ret["alpha_sample"] = alpha_sample;
  ret["alpha_p_sample"] = alpha_p_sample;
  ret["lambda_sample"] = lambda_sample;
  ret["sigmaSq_Sigma_sample"] = sigmaSq_Sigma_sample;
  ret["sigmaSq_sample"] = sigmaSq_sample;
  
  return ret;
}
