#ifndef __TAU_H__
#define __TAU_H__

#include <Rcpp.h>
#include <RcppEigen.h>
#include "basic.h"
#include <random>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// [[Rcpp::depends(RcppEigen)]]

// Library Functions
MatrixXd add_step(MatrixXd eff_coeff, const double V_tau, const VectorXd& nu, const VectorXd& lambda, const MatrixXd& Sigma, const MatrixXd& C_mat, const double theta_gamma, const MatrixXd& M, const MatrixXd& beta0, const VectorXd& treat, const MatrixXd& X, const MatrixXd& B, const MatrixXd& Sigma_inv, const double eta);
MatrixXd delete_step(MatrixXd eff_coeff, const double V_tau, const VectorXd& nu, const VectorXd& lambda, const MatrixXd& Sigma, const MatrixXd& C_mat, const double theta_gamma, const MatrixXd& M, const MatrixXd& beta0, const VectorXd& treat, const MatrixXd& X, const MatrixXd& B, const MatrixXd& Sigma_inv, const double eta);
MatrixXd delete_step2(MatrixXd eff_coeff, const double V_tau, const VectorXd& nu, const VectorXd& lambda, const MatrixXd& Sigma, const MatrixXd& C_mat, const double theta_gamma, const MatrixXd& M, const MatrixXd& beta0, const VectorXd& treat, const MatrixXd& X, const MatrixXd& B, const MatrixXd& Sigma_inv, const double eta);
MatrixXd swap_step(MatrixXd eff_coeff, const double V_tau, const VectorXd& nu, const VectorXd& lambda, const MatrixXd& Sigma, const MatrixXd& C_mat, const double theta_gamma, const MatrixXd& M, const MatrixXd& beta0, const VectorXd& treat, const MatrixXd& X, const MatrixXd& B, const MatrixXd& Sigma_inv, const double eta);
MatrixXd swap_step2(MatrixXd eff_coeff, const double V_tau, const VectorXd& nu, const VectorXd& lambda, const MatrixXd& Sigma, const MatrixXd& C_mat, const double theta_gamma, const MatrixXd& M, const MatrixXd& beta0, const VectorXd& treat, const MatrixXd& X, const MatrixXd& B, const MatrixXd& Sigma_inv, const double eta);

MatrixXd refining_step (const MatrixXd& M, const MatrixXd& beta0, const VectorXd& treat, const MatrixXd& X, const MatrixXd& B, const MatrixXd& Sigma_inv, const MatrixXd& Sigma, const VectorXd& nu, MatrixXd eff_coeff);

MatrixXd update_tau (MatrixXd eff_coeff, const double V_tau, const VectorXd& nu, const VectorXd& lambda, const MatrixXd& Sigma, const MatrixXd& C_mat, const double theta_gamma, const MatrixXd& M, const MatrixXd& beta0, const VectorXd& treat, const MatrixXd& X, const MatrixXd& B, const MatrixXd& Sigma_inv, const double eta);
MatrixXd update_tau2 (MatrixXd eff_coeff, const double V_tau, const VectorXd& nu, const VectorXd& lambda, const MatrixXd& Sigma, const MatrixXd& C_mat, const double theta_gamma, const MatrixXd& M, const MatrixXd& beta0, const VectorXd& treat, const MatrixXd& X, const MatrixXd& B, const MatrixXd& Sigma_inv, const double eta);

// Functions Define
// 1. add update
MatrixXd add_step(MatrixXd eff_coeff,
                  const double V_tau,
                  const VectorXd& nu,
                  const VectorXd& lambda,
                  const MatrixXd& Sigma,
                  const MatrixXd& C_mat,
                  const double theta_gamma,
                  const MatrixXd& M,
                  const MatrixXd& beta0,
                  const VectorXd& treat,
                  const MatrixXd& X,
                  const MatrixXd& B,
                  const MatrixXd& Sigma_inv,
                  const double eta){
    int q = M.cols();

    VectorXd tau = eff_coeff.col(0);
    VectorXd gamma = eff_coeff.col(1);

    double g_gamma = gamma.sum();
    int l_idx = random_zero_index(gamma);

    VectorXd tau_star = tau;
    tau_star(l_idx) = Rcpp::rnorm(1, tau(l_idx), sqrt(V_tau))(0);
    VectorXd gamma_star = gamma;
    gamma_star(l_idx) = 1.0;

    // add prior ratio
    double log_prior_part1 = log_normal_dist(tau_star(l_idx), Sigma(l_idx, l_idx) * pow(nu(l_idx), 2));
    double log_prior_part2 = (C_mat.row(l_idx) * gamma);
    double log_prior_part3 = theta_gamma + eta * log_prior_part2;
    double log_prior_ratio = log_prior_part1 + log_prior_part3;

    // add proposal ratio
    double log_proposal_part1 = -log_normal_dist(tau_star(l_idx), V_tau);
    double log_proposal_ratio = log_proposal_part1 + log((q - g_gamma) / (g_gamma + 1.0));

    // add likelihood ratio
    MatrixXd res_m_star = M - beta0 - (treat * tau_star.transpose()) - (X * B);
    MatrixXd res_m = M - beta0 - (treat * tau.transpose()) - (X * B);

    VectorXd num_vec = (res_m_star * Sigma_inv * res_m_star.transpose()).diagonal();
    VectorXd den_vec = (res_m * Sigma_inv * res_m.transpose()).diagonal();

    double num_m = num_vec.sum();
    double den_m = den_vec.sum();

    double log_likelihood_ratio = (den_m - num_m) / 2.0;

    // add accept/reject
    double log_accept_rate_tau = log_prior_ratio + log_proposal_ratio + log_likelihood_ratio;

    double log_u = log(Rcpp::runif(1)(0));

    if (log_accept_rate_tau > log_u){
        eff_coeff.col(0) = tau_star;
        eff_coeff.col(1) = gamma_star;
    }
    return eff_coeff;
}

// 2. delete update
MatrixXd delete_step(MatrixXd eff_coeff,
                     const double V_tau,
                     const VectorXd& nu,
                     const VectorXd& lambda,
                     const MatrixXd& Sigma,
                     const MatrixXd& C_mat,
                     const double theta_gamma,
                     const MatrixXd& M,
                     const MatrixXd& beta0,
                     const VectorXd& treat,
                     const MatrixXd& X,
                     const MatrixXd& B,
                     const MatrixXd& Sigma_inv,
                     const double eta){
  int q = M.cols();
  
  VectorXd tau   = eff_coeff.col(0);
  VectorXd gamma = eff_coeff.col(1);
  
  double g_gamma = gamma.sum();
  int m_idx = random_nonzero_index(gamma);
  
  VectorXd tau_star   = tau;
  tau_star(m_idx)   = 0.0;
  VectorXd gamma_star = gamma;
  gamma_star(m_idx) = 0.0;
  
  // delete prior ratio
  double log_prior_part1 = -log_normal_dist(tau(m_idx), Sigma(m_idx, m_idx) * pow(nu(m_idx), 2));
  double log_prior_part2 = (C_mat.row(m_idx) * gamma);
  double log_prior_part3 = theta_gamma + eta * log_prior_part2;
  double log_prior_ratio = log_prior_part1 - log_prior_part3;

  // delete proposal ratio
  double log_proposal_par1 = log_normal_dist(tau(m_idx), V_tau);
  double log_proposal_ratio = log_proposal_par1 + log(g_gamma/(q-g_gamma+1.0));

  // delete likelihood ratio
  MatrixXd res_m_star = M - beta0 - (treat * tau_star.transpose()) - (X * B);
  MatrixXd res_m      = M - beta0 - (treat * tau.transpose()     ) - (X * B);

  VectorXd num_vec = (res_m_star * Sigma_inv * res_m_star.transpose()).diagonal();
  VectorXd den_vec = (res_m      * Sigma_inv * res_m.transpose()     ).diagonal();

  double num_m = num_vec.sum();
  double den_m = den_vec.sum();

  double log_likelihood_ratio = (den_m - num_m) / 2.0;
  
  // delete accept/reject
  double log_accept_rate_tau = log_prior_ratio + log_proposal_ratio + log_likelihood_ratio;

  double log_u = log(Rcpp::runif(1)(0));

  if (log_accept_rate_tau > log_u){
      eff_coeff.col(0) = tau_star;
      eff_coeff.col(1) = gamma_star;
  }
  return eff_coeff;
}

// 3. swap update
MatrixXd swap_step(MatrixXd eff_coeff,
                   const double V_tau,
                   const VectorXd& nu,
                   const VectorXd& lambda,
                   const MatrixXd& Sigma,
                   const MatrixXd& C_mat,
                   const double theta_gamma,
                   const MatrixXd& M,
                   const MatrixXd& beta0,
                   const VectorXd& treat,
                   const MatrixXd& X,
                   const MatrixXd& B,
                   const MatrixXd& Sigma_inv,
                   const double eta){
  VectorXd tau   = eff_coeff.col(0);
  VectorXd gamma = eff_coeff.col(1);
  
  int l_idx = random_zero_index(gamma);
  int m_idx = random_nonzero_index(gamma);

  VectorXd tau_star   = tau;
  tau_star(l_idx) = Rcpp::rnorm(1, tau(l_idx), sqrt(V_tau))(0);
  tau_star(m_idx) = 0.0;
  VectorXd gamma_star = gamma;
  gamma_star(l_idx) = 1.0;
  gamma_star(m_idx) = 0.0;

  // swap prior ratio
  double num_part_prior1 = log_normal_dist(tau_star(l_idx), Sigma(l_idx, l_idx) * pow(nu(l_idx), 2));
  double num_part_priot2 = (C_mat.row(l_idx) * gamma);
  double num_part_prior3 = theta_gamma + eta * num_part_priot2;

  double den_part_prior1 = log_normal_dist(tau(m_idx), Sigma(m_idx, m_idx) * pow(nu(m_idx), 2));
  double den_part_priot2 = (C_mat.row(m_idx) * gamma);
  double den_part_prior3 = theta_gamma + eta * den_part_priot2;

  double log_prior_ratio = num_part_prior1 + num_part_prior3 - den_part_prior1 - den_part_prior3;

  // swap proposal ratio
  double log_proposal_part1 = log_normal_dist(tau(m_idx), V_tau);
  double log_proposal_part2 = log_normal_dist(tau_star(l_idx), V_tau);
  double log_proposal_ratio = log_proposal_part1 - log_proposal_part2;

  // swap likelihood ratio
  MatrixXd res_m_star = M - beta0 - (treat * tau_star.transpose()) - (X * B);
  MatrixXd res_m      = M - beta0 - (treat * tau.transpose()     ) - (X * B);

  VectorXd num_vec = (res_m_star * Sigma_inv * res_m_star.transpose()).diagonal();
  VectorXd den_vec = (res_m      * Sigma_inv * res_m.transpose()     ).diagonal();

  double num_m = num_vec.sum();
  double den_m = den_vec.sum();

  double log_likelihood_ratio = (den_m - num_m) / 2.0;
  
  // swap accept/reject
  double log_accept_rate_tau = log_prior_ratio + log_proposal_ratio + log_likelihood_ratio;

  double log_u = log(Rcpp::runif(1)(0));

  if (log_accept_rate_tau > log_u){
      eff_coeff.col(0) = tau_star;
      eff_coeff.col(1) = gamma_star;
  }
  return eff_coeff;
}

// 2-1. delete update
MatrixXd delete_step2(MatrixXd eff_coeff,
                     const double V_tau,
                     const VectorXd& nu,
                     const VectorXd& lambda,
                     const MatrixXd& Sigma,
                     const MatrixXd& C_mat,
                     const double theta_gamma,
                     const MatrixXd& M,
                     const MatrixXd& beta0,
                     const VectorXd& treat,
                     const MatrixXd& X,
                     const MatrixXd& B,
                     const MatrixXd& Sigma_inv,
                     const double eta){
  int q = M.cols();
  
  VectorXd tau   = eff_coeff.col(0);
  VectorXd gamma = eff_coeff.col(1);
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  
  double g_gamma = gamma.sum();
  int m_idx = random_nonzero_index(gamma);
  
  VectorXd tau_star   = tau;
  tau_star(m_idx)   = 0.0;
  VectorXd gamma_star = gamma;
  gamma_star(m_idx) = 0.0;
  VectorXd delta_star = delta;
  delta_star(m_idx) = 0.0;
  VectorXd omega_star = omega;
  omega_star(m_idx) = 0.0;
  
  // delete prior ratio
  double log_prior_part1 = -log_normal_dist(tau(m_idx), Sigma(m_idx, m_idx) * pow(nu(m_idx), 2));
  double log_prior_part2 = (C_mat.row(m_idx) * gamma);
  double log_prior_part3 = theta_gamma + eta * log_prior_part2;
  double log_prior_ratio = log_prior_part1 - log_prior_part3;
  
  // delete proposal ratio
  double log_proposal_par1 = log_normal_dist(tau(m_idx), V_tau);
  double log_proposal_ratio = log_proposal_par1 + log(g_gamma/(q-g_gamma+1.0));
  
  // delete likelihood ratio
  MatrixXd res_m_star = M - beta0 - (treat * tau_star.transpose()) - (X * B);
  MatrixXd res_m      = M - beta0 - (treat * tau.transpose()     ) - (X * B);
  
  VectorXd num_vec = (res_m_star * Sigma_inv * res_m_star.transpose()).diagonal();
  VectorXd den_vec = (res_m      * Sigma_inv * res_m.transpose()     ).diagonal();
  
  double num_m = num_vec.sum();
  double den_m = den_vec.sum();
  
  double log_likelihood_ratio = (den_m - num_m) / 2.0;
  
  // delete accept/reject
  double log_accept_rate_tau = log_prior_ratio + log_proposal_ratio + log_likelihood_ratio;
  
  double log_u = log(Rcpp::runif(1)(0));
  
  if (log_accept_rate_tau > log_u){
    eff_coeff.col(0) = tau_star;
    eff_coeff.col(1) = gamma_star;
    eff_coeff.col(2) = delta_star;
    eff_coeff.col(3) = omega_star;
  }
  return eff_coeff;
}

// 3-1. swap update
MatrixXd swap_step2(MatrixXd eff_coeff,
                   const double V_tau,
                   const VectorXd& nu,
                   const VectorXd& lambda,
                   const MatrixXd& Sigma,
                   const MatrixXd& C_mat,
                   const double theta_gamma,
                   const MatrixXd& M,
                   const MatrixXd& beta0,
                   const VectorXd& treat,
                   const MatrixXd& X,
                   const MatrixXd& B,
                   const MatrixXd& Sigma_inv,
                   const double eta){
  VectorXd tau   = eff_coeff.col(0);
  VectorXd gamma = eff_coeff.col(1);
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  
  int l_idx = random_zero_index(gamma);
  int m_idx = random_nonzero_index(gamma);
  
  VectorXd tau_star   = tau;
  tau_star(l_idx) = Rcpp::rnorm(1, tau(l_idx), sqrt(V_tau))(0);
  tau_star(m_idx) = 0.0;
  VectorXd gamma_star = gamma;
  gamma_star(l_idx) = 1.0;
  gamma_star(m_idx) = 0.0;
  VectorXd delta_star = delta;
  delta_star(m_idx) = 0.0;
  VectorXd omega_star = omega;
  omega_star(m_idx) = 0.0;
  
  // swap prior ratio
  double num_part_prior1 = log_normal_dist(tau_star(l_idx), Sigma(l_idx, l_idx) * pow(nu(l_idx), 2));
  double num_part_priot2 = (C_mat.row(l_idx) * gamma);
  double num_part_prior3 = theta_gamma + eta * num_part_priot2;
  
  double den_part_prior1 = log_normal_dist(tau(m_idx), Sigma(m_idx, m_idx) * pow(nu(m_idx), 2));
  double den_part_priot2 = (C_mat.row(m_idx) * gamma);
  double den_part_prior3 = theta_gamma + eta * den_part_priot2;
  
  double log_prior_ratio = num_part_prior1 + num_part_prior3 - den_part_prior1 - den_part_prior3;
  
  // swap proposal ratio
  double log_proposal_part1 = log_normal_dist(tau(m_idx), V_tau);
  double log_proposal_part2 = log_normal_dist(tau_star(l_idx), V_tau);
  double log_proposal_ratio = log_proposal_part1 - log_proposal_part2;
  
  // swap likelihood ratio
  MatrixXd res_m_star = M - beta0 - (treat * tau_star.transpose()) - (X * B);
  MatrixXd res_m      = M - beta0 - (treat * tau.transpose()     ) - (X * B);
  
  VectorXd num_vec = (res_m_star * Sigma_inv * res_m_star.transpose()).diagonal();
  VectorXd den_vec = (res_m      * Sigma_inv * res_m.transpose()     ).diagonal();
  
  double num_m = num_vec.sum();
  double den_m = den_vec.sum();
  
  double log_likelihood_ratio = (den_m - num_m) / 2.0;
  
  // swap accept/reject
  double log_accept_rate_tau = log_prior_ratio + log_proposal_ratio + log_likelihood_ratio;
  
  double log_u = log(Rcpp::runif(1)(0));
  
  if (log_accept_rate_tau > log_u){
    eff_coeff.col(0) = tau_star;
    eff_coeff.col(1) = gamma_star;
    eff_coeff.col(2) = delta_star;
    eff_coeff.col(3) = omega_star;
  }
  return eff_coeff;
}

// 4. refine step
MatrixXd refining_step (const MatrixXd& M,
                        const MatrixXd& beta0,
                        const VectorXd& treat,
                        const MatrixXd& X,
                        const MatrixXd& B,
                        const MatrixXd& Sigma_inv,
                        const MatrixXd& Sigma,
                        const VectorXd& nu,
                        MatrixXd eff_coeff){
  VectorXd gamma      = eff_coeff.col(1);
  VectorXd update_idx = nonzero_index(gamma);
  int update_size = update_idx.rows();

  if(update_size > 0){
    MatrixXd new_M               = MatrixXd::Zero(M.rows(), update_size);
    MatrixXd new_beta0           = MatrixXd::Zero(beta0.rows(), update_size);
    MatrixXd new_B               = MatrixXd::Zero(B.rows(), update_size);
    MatrixXd new_Sigma_inv       = MatrixXd::Zero(update_size, update_size);
    MatrixXd new_prior_Sigma_inv = MatrixXd::Zero(update_size, update_size);
    
    for(int i=0; i<update_size; i++){
      int idx = update_idx(i);
      new_M.col(i)     = M.col(idx);
      new_beta0.col(i) = beta0.col(idx);
      new_B.col(i)     = B.col(idx);
      for(int j=0; j<update_size; j++){
        int idx_j = update_idx(j);
        new_Sigma_inv(i,j) = Sigma_inv(idx, idx_j);
      }
      new_prior_Sigma_inv(i,i) = 1.0/(pow(nu(idx),2)*Sigma(idx,idx));
    }
    
    double new_treat = treat.squaredNorm();
    
    MatrixXd update_Sigma_part = new_treat * new_Sigma_inv + new_prior_Sigma_inv;
    MatrixXd update_Sigma = update_Sigma_part.inverse();
    
    MatrixXd new_res_m = new_M - new_beta0 - X * new_B;
    
    VectorXd update_mu_part = new_Sigma_inv * new_res_m.transpose() * treat;
    VectorXd update_mu = update_Sigma * update_mu_part;

    VectorXd z = sampleMultivariateNormal(update_mu, update_Sigma, 1).row(0);
    int idxx;
    for(int k=0; k<update_size; k++){
      idxx = update_idx(k);
      eff_coeff(idxx,0) = z(k);
    }
  }
  return eff_coeff;
}

// 5. tau_update
MatrixXd update_tau (MatrixXd eff_coeff,
                     const double V_tau,
                     const VectorXd& nu,
                     const VectorXd& lambda,
                     const MatrixXd& Sigma,
                     const MatrixXd& C_mat,
                     const double theta_gamma,
                     const MatrixXd& M,
                     const MatrixXd& beta0,
                     const VectorXd& treat,
                     const MatrixXd& X,
                     const MatrixXd& B,
                     const MatrixXd& Sigma_inv,
                     const double eta){
  VectorXd tau   = eff_coeff.col(0);
  VectorXd gamma = eff_coeff.col(1);
  int q = tau.rows(), g_gamma = gamma.sum(), move_1;

  if(g_gamma == 0){
    move_1 = 1;
  }else if(g_gamma == q){
    move_1 = 2;
  }else{
    move_1 = ran_sample(3);
  }
  
  MatrixXd result1 = MatrixXd::Zero(q, 4);
  MatrixXd result2 = MatrixXd::Zero(q, 4);
  
  switch(move_1){
  case 1: // add step
    result1 = add_step(eff_coeff, V_tau, nu, lambda, Sigma, C_mat, theta_gamma, M, beta0, treat, X, B, Sigma_inv, eta);
    break;
  case 2: // delete step
    result1 = delete_step(eff_coeff, V_tau, nu, lambda, Sigma, C_mat, theta_gamma, M, beta0, treat, X, B, Sigma_inv, eta);
    break;
  case 3: // swap step
    result1 = swap_step(eff_coeff, V_tau, nu, lambda, Sigma, C_mat, theta_gamma, M, beta0, treat, X, B, Sigma_inv, eta);
    break;
  }
  eff_coeff = result1;
  
  // refining step
  result2 = refining_step(M, beta0, treat, X, B, Sigma_inv, Sigma, nu, eff_coeff);
  eff_coeff = result2;
  
  return eff_coeff;
}

// 5-1. tau_update
MatrixXd update_tau2 (MatrixXd eff_coeff,
                     const double V_tau,
                     const VectorXd& nu,
                     const VectorXd& lambda,
                     const MatrixXd& Sigma,
                     const MatrixXd& C_mat,
                     const double theta_gamma,
                     const MatrixXd& M,
                     const MatrixXd& beta0,
                     const VectorXd& treat,
                     const MatrixXd& X,
                     const MatrixXd& B,
                     const MatrixXd& Sigma_inv,
                     const double eta){
  VectorXd tau   = eff_coeff.col(0);
  VectorXd gamma = eff_coeff.col(1);
  int q = tau.rows(), g_gamma = gamma.sum(), move_1;
  
  if(g_gamma == 0){
    move_1 = 1;
  }else if(g_gamma == q){
    move_1 = 2;
  }else{
    move_1 = ran_sample(3);
  }
  
  MatrixXd result1 = MatrixXd::Zero(q, 4);
  MatrixXd result2 = MatrixXd::Zero(q, 4);
  
  switch(move_1){
  case 1: // add step
    result1 = add_step(eff_coeff, V_tau, nu, lambda, Sigma, C_mat, theta_gamma, M, beta0, treat, X, B, Sigma_inv, eta);
    break;
  case 2: // delete step
    result1 = delete_step2(eff_coeff, V_tau, nu, lambda, Sigma, C_mat, theta_gamma, M, beta0, treat, X, B, Sigma_inv, eta);
    break;
  case 3: // swap step
    result1 = swap_step2(eff_coeff, V_tau, nu, lambda, Sigma, C_mat, theta_gamma, M, beta0, treat, X, B, Sigma_inv, eta);
    break;
  }
  eff_coeff = result1;
  
  // refining step
  result2 = refining_step(M, beta0, treat, X, B, Sigma_inv, Sigma, nu, eff_coeff);
  eff_coeff = result2;
  
  return eff_coeff;
}

#endif