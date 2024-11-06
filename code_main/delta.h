#ifndef __DELTA_H__
#define __DELTA_H__

#include <Rcpp.h>
#include <RcppEigen.h>
#include "basic.h"
#include <random>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// [[Rcpp::depends(RcppEigen)]]

// Library Functions
MatrixXd add_step1(MatrixXd eff_coeff, const double V_delta, const VectorXd& psi, const double sigmaSq, const double theta_omega, const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& alpha, const double alpha_p, const VectorXd& treat, const MatrixXd& X);
MatrixXd delete_step1(MatrixXd eff_coeff, const double V_delta, const VectorXd& psi, const double sigmaSq, const double theta_omega, const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& alpha, const double alpha_p, const VectorXd& treat, const MatrixXd& X);
MatrixXd swap_step1(MatrixXd eff_coeff, const double V_delta, const VectorXd& psi, const double sigmaSq, const double theta_omega, const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& alpha, const double alpha_p, const VectorXd& treat, const MatrixXd& X);
MatrixXd refining_step1(MatrixXd eff_coeff, const VectorXd& psi, const double sigmaSq, const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& alpha, const double alpha_p, const VectorXd& treat, const MatrixXd& X);
MatrixXd add_step2(MatrixXd eff_coeff, const double V_delta, const VectorXd& psi, const double sigmaSq, const double theta_omega, const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& alpha, const double alpha_p, const VectorXd& treat, const MatrixXd& X);
MatrixXd delete_step2(MatrixXd eff_coeff, const double V_delta, const VectorXd& psi, const double sigmaSq, const double theta_omega, const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& alpha, const double alpha_p, const VectorXd& treat, const MatrixXd& X);
MatrixXd swap_step2(MatrixXd eff_coeff, const double V_delta, const VectorXd& psi, const double sigmaSq, const double theta_omega, const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& alpha, const double alpha_p, const VectorXd& treat, const MatrixXd& X);

MatrixXd update_delta(MatrixXd eff_coeff, const double V_delta, const VectorXd& psi, const double sigmaSq, const double theta_omega, const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& alpha, const double alpha_p, const VectorXd& treat, const MatrixXd& X);
MatrixXd update_delta2(MatrixXd eff_coeff, const double V_delta, const VectorXd& psi, const double sigmaSq, const double theta_omega, const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& alpha, const double alpha_p, const VectorXd& treat, const MatrixXd& X);

// Functions Define
// 1. add update
MatrixXd add_step1(MatrixXd eff_coeff,
                   const double V_delta,
                   const VectorXd& psi,
                   const double sigmaSq,
                   const double theta_omega,
                   const VectorXd& y,
                   const VectorXd& alpha0,
                   const MatrixXd& M,
                   const VectorXd& alpha,
                   const double alpha_p,
                   const VectorXd& treat,
                   const MatrixXd& X){
  int q = M.cols();
  
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  
  double g_omega = omega.sum();
  int l_idx = random_zero_index(omega);

  VectorXd delta_star = delta;
  delta_star(l_idx) = Rcpp::rnorm(1, delta(l_idx), sqrt(V_delta))(0);
  VectorXd omega_star = omega;
  omega_star(l_idx) = 1.0;
  
  // add prior ratio
  double log_prior_part1 = log_normal_dist(delta_star(l_idx), sigmaSq * pow(psi(l_idx), 2));
  double log_prior_ratio = log_prior_part1 + log(theta_omega/(1.0-theta_omega));
  
  // add proposal ratio
  double log_proposal_part1 = -log_normal_dist(delta_star(l_idx), V_delta);
  double log_proposal_ratio = log_proposal_part1 + log((q-g_omega)/(g_omega+1.0));

  // add likelihood ratio
  VectorXd res_y_star = y - alpha0 - (M * delta_star) - (X * alpha) - (alpha_p * treat);
  VectorXd res_y      = y - alpha0 - (M * delta     ) - (X * alpha) - (alpha_p * treat);

  double num_y = res_y_star.squaredNorm();
  double den_y = res_y.squaredNorm();
  
  double log_likelihood_ratio = (den_y - num_y) / (2.0 * sigmaSq);
  
  // add accept/reject
  double log_accept_rate_delta = log_prior_ratio + log_proposal_ratio + log_likelihood_ratio;
  double log_u = log(Rcpp::runif(1)(0));
  
  if(log_accept_rate_delta > log_u){
    eff_coeff.col(2) = delta_star;
    eff_coeff.col(3) = omega_star;
  }
  return eff_coeff;
}

// 2. delete update
MatrixXd delete_step1(MatrixXd eff_coeff,
                      const double V_delta,
                      const VectorXd& psi,
                      const double sigmaSq,
                      const double theta_omega,
                      const VectorXd& y,
                      const VectorXd& alpha0,
                      const MatrixXd& M,
                      const VectorXd& alpha,
                      const double alpha_p,
                      const VectorXd& treat,
                      const MatrixXd& X){
  int q = M.cols();
  
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  
  double g_omega = omega.sum();
  int m_idx = random_nonzero_index(omega);
  
  VectorXd delta_star = delta;
  delta_star(m_idx) = 0.0;
  VectorXd omega_star = omega;
  omega_star(m_idx) = 0.0;
  
  // delete prior ratio
  double log_prior_part1 = -log_normal_dist(delta(m_idx), sigmaSq * pow(psi(m_idx), 2));
  double log_prior_ratio = log_prior_part1 + log((1.0-theta_omega)/theta_omega);

  // delete proposal ratio
  double log_proposal_part1 = log_normal_dist(delta(m_idx), V_delta);
  double log_proposal_ratio = log_proposal_part1 + log(g_omega/(q-g_omega+1.0));

  // delete likelihood ratio
  VectorXd res_y_star = y - alpha0 - (M * delta_star) - (X * alpha) - (alpha_p * treat);
  VectorXd res_y      = y - alpha0 - (M * delta     ) - (X * alpha) - (alpha_p * treat);

  double num_y = res_y_star.squaredNorm();
  double den_y = res_y.squaredNorm();
  
  double log_likelihood_ratio = (den_y - num_y) / (2.0 * sigmaSq);
  
  // add accept/reject
  double log_accept_rate_delta = log_prior_ratio + log_proposal_ratio + log_likelihood_ratio;
  double log_u = log(Rcpp::runif(1)(0));
  
  if(log_accept_rate_delta > log_u){
    eff_coeff.col(2) = delta_star;
    eff_coeff.col(3) = omega_star;
  }
  return eff_coeff;
}

// 3. swap update
MatrixXd swap_step1(MatrixXd eff_coeff,
                    const double V_delta,
                    const VectorXd& psi,
                    const double sigmaSq,
                    const double theta_omega,
                    const VectorXd& y,
                    const VectorXd& alpha0,
                    const MatrixXd& M,
                    const VectorXd& alpha,
                    const double alpha_p,
                    const VectorXd& treat,
                    const MatrixXd& X){
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  int l_idx = random_zero_index(omega);
  int m_idx = random_nonzero_index(omega);

  VectorXd delta_star = delta;
  delta_star(l_idx) = Rcpp::rnorm(1, delta(l_idx), sqrt(V_delta))(0);
  delta_star(m_idx) = 0.0;
  VectorXd omega_star = omega;
  omega_star(l_idx) = 1.0;
  omega_star(m_idx) = 0.0;
  
  // swap prior ratio
  double num_part_prior = log_normal_dist(delta_star(l_idx), sigmaSq * pow(psi(l_idx), 2));
  double den_part_prior = log_normal_dist(delta(m_idx), sigmaSq * pow(psi(m_idx), 2));
  double log_prior_ratio = num_part_prior - den_part_prior;

  // swap proposal ratio
  double num_part_proposal = log_normal_dist(delta(m_idx), V_delta);
  double den_part_proposal = log_normal_dist(delta_star(l_idx), V_delta);
  double log_proposal_ratio = num_part_proposal - den_part_proposal;

  // swap likelihood ratio
  VectorXd res_y_star = y - alpha0 - (M * delta_star) - (X * alpha) - (alpha_p * treat);
  VectorXd res_y      = y - alpha0 - (M * delta     ) - (X * alpha) - (alpha_p * treat);

  double num_y = res_y_star.squaredNorm();
  double den_y = res_y.squaredNorm();
  
  double log_likelihood_ratio = (den_y - num_y) / (2.0 * sigmaSq);
  
  // add accept/reject
  double log_accept_rate_delta = log_prior_ratio + log_proposal_ratio + log_likelihood_ratio;
  double log_u = log(Rcpp::runif(1)(0));
  
  if(log_accept_rate_delta > log_u){
    eff_coeff.col(2) = delta_star;
    eff_coeff.col(3) = omega_star;
  }
  return eff_coeff;
}

// 1-1. add update
MatrixXd add_step2(MatrixXd eff_coeff,
                   const double V_delta,
                   const VectorXd& psi,
                   const double sigmaSq,
                   const double theta_omega,
                   const VectorXd& y,
                   const VectorXd& alpha0,
                   const MatrixXd& M,
                   const VectorXd& alpha,
                   const double alpha_p,
                   const VectorXd& treat,
                   const MatrixXd& X){
  int q = M.cols();
  
  VectorXd gamma = eff_coeff.col(1);
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  
  double g_omega = omega.sum();
  int l_idx = random_zero_index_given(omega, gamma);
  
  VectorXd delta_star = delta;
  delta_star(l_idx) = Rcpp::rnorm(1, delta(l_idx), sqrt(V_delta))(0);
  VectorXd omega_star = omega;
  omega_star(l_idx) = 1.0;
  
  // add prior ratio
  double log_prior_part1 = log_normal_dist(delta_star(l_idx), sigmaSq * pow(psi(l_idx), 2));
  double log_prior_ratio = log_prior_part1 + log(theta_omega/(1.0-theta_omega));
  
  // add proposal ratio
  double log_proposal_part1 = -log_normal_dist(delta_star(l_idx), V_delta);
  double log_proposal_ratio = log_proposal_part1 + log((q-g_omega)/(g_omega+1.0));
  
  // add likelihood ratio
  VectorXd res_y_star = y - alpha0 - (M * delta_star) - (X * alpha) - (alpha_p * treat);
  VectorXd res_y      = y - alpha0 - (M * delta     ) - (X * alpha) - (alpha_p * treat);
  
  double num_y = res_y_star.squaredNorm();
  double den_y = res_y.squaredNorm();
  
  double log_likelihood_ratio = (den_y - num_y) / (2.0 * sigmaSq);
  
  // add accept/reject
  double log_accept_rate_delta = log_prior_ratio + log_proposal_ratio + log_likelihood_ratio;
  double log_u = log(Rcpp::runif(1)(0));
  
  if(log_accept_rate_delta > log_u){
    eff_coeff.col(2) = delta_star;
    eff_coeff.col(3) = omega_star;
  }
  return eff_coeff;
}

// 2-1. delete update
MatrixXd delete_step2(MatrixXd eff_coeff,
                      const double V_delta,
                      const VectorXd& psi,
                      const double sigmaSq,
                      const double theta_omega,
                      const VectorXd& y,
                      const VectorXd& alpha0,
                      const MatrixXd& M,
                      const VectorXd& alpha,
                      const double alpha_p,
                      const VectorXd& treat,
                      const MatrixXd& X){
  int q = M.cols();
  
  VectorXd gamma = eff_coeff.col(1);
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  
  double g_omega = omega.sum();
  int m_idx = random_nonzero_index_given(omega, gamma);
  
  VectorXd delta_star = delta;
  delta_star(m_idx) = 0.0;
  VectorXd omega_star = omega;
  omega_star(m_idx) = 0.0;
  
  // delete prior ratio
  double log_prior_part1 = -log_normal_dist(delta(m_idx), sigmaSq * pow(psi(m_idx), 2));
  double log_prior_ratio = log_prior_part1 + log((1.0-theta_omega)/theta_omega);
  
  // delete proposal ratio
  double log_proposal_part1 = log_normal_dist(delta(m_idx), V_delta);
  double log_proposal_ratio = log_proposal_part1 + log(g_omega/(q-g_omega+1.0));
  
  // delete likelihood ratio
  VectorXd res_y_star = y - alpha0 - (M * delta_star) - (X * alpha) - (alpha_p * treat);
  VectorXd res_y      = y - alpha0 - (M * delta     ) - (X * alpha) - (alpha_p * treat);
  
  double num_y = res_y_star.squaredNorm();
  double den_y = res_y.squaredNorm();
  
  double log_likelihood_ratio = (den_y - num_y) / (2.0 * sigmaSq);
  
  // add accept/reject
  double log_accept_rate_delta = log_prior_ratio + log_proposal_ratio + log_likelihood_ratio;
  double log_u = log(Rcpp::runif(1)(0));
  
  if(log_accept_rate_delta > log_u){
    eff_coeff.col(2) = delta_star;
    eff_coeff.col(3) = omega_star;
  }
  return eff_coeff;
}

// 3-1. swap update
MatrixXd swap_step2(MatrixXd eff_coeff,
                    const double V_delta,
                    const VectorXd& psi,
                    const double sigmaSq,
                    const double theta_omega,
                    const VectorXd& y,
                    const VectorXd& alpha0,
                    const MatrixXd& M,
                    const VectorXd& alpha,
                    const double alpha_p,
                    const VectorXd& treat,
                    const MatrixXd& X){
  
  VectorXd gamma = eff_coeff.col(1);
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  
  int l_idx = random_zero_index_given(omega, gamma);
  int m_idx = random_nonzero_index_given(omega, gamma);
  
  VectorXd delta_star = delta;
  delta_star(l_idx) = Rcpp::rnorm(1, delta(l_idx), sqrt(V_delta))(0);
  delta_star(m_idx) = 0.0;
  VectorXd omega_star = omega;
  omega_star(l_idx) = 1.0;
  omega_star(m_idx) = 0.0;
  
  // swap prior ratio
  double num_part_prior = log_normal_dist(delta_star(l_idx), sigmaSq * pow(psi(l_idx), 2));
  double den_part_prior = log_normal_dist(delta(m_idx), sigmaSq * pow(psi(m_idx), 2));
  double log_prior_ratio = num_part_prior - den_part_prior;
  
  // swap proposal ratio
  double num_part_proposal = log_normal_dist(delta(m_idx), V_delta);
  double den_part_proposal = log_normal_dist(delta_star(l_idx), V_delta);
  double log_proposal_ratio = num_part_proposal - den_part_proposal;
  
  // swap likelihood ratio
  VectorXd res_y_star = y - alpha0 - (M * delta_star) - (X * alpha) - (alpha_p * treat);
  VectorXd res_y      = y - alpha0 - (M * delta     ) - (X * alpha) - (alpha_p * treat);
  
  double num_y = res_y_star.squaredNorm();
  double den_y = res_y.squaredNorm();
  
  double log_likelihood_ratio = (den_y - num_y) / (2.0 * sigmaSq);
  
  // add accept/reject
  double log_accept_rate_delta = log_prior_ratio + log_proposal_ratio + log_likelihood_ratio;
  double log_u = log(Rcpp::runif(1)(0));
  
  if(log_accept_rate_delta > log_u){
    eff_coeff.col(2) = delta_star;
    eff_coeff.col(3) = omega_star;
  }
  return eff_coeff;
}

// 4. refine step
MatrixXd refining_step1(MatrixXd eff_coeff,
                        const VectorXd& psi,
                        const double sigmaSq,
                        const VectorXd& y,
                        const VectorXd& alpha0,
                        const MatrixXd& M,
                        const VectorXd& alpha,
                        const double alpha_p,
                        const VectorXd& treat,
                        const MatrixXd& X){
  VectorXd omega      = eff_coeff.col(3);
  VectorXd update_idx = nonzero_index(omega);
  int update_size = update_idx.rows();
  
  if(update_size > 0){
    MatrixXd new_M = MatrixXd::Zero(M.rows(), update_size);
    MatrixXd new_prior_Sigma_inv = MatrixXd::Zero(update_size, update_size);
    
    for(int i=0; i<update_size; i++){
      int idx = update_idx(i);
      new_M.col(i) = M.col(idx);
      new_prior_Sigma_inv(i,i) = 1.0 / (pow(psi(idx),2)*sigmaSq);
    }
    
    MatrixXd update_Sigma_part = (1.0/sigmaSq) * new_M.transpose() * new_M + new_prior_Sigma_inv;
    MatrixXd update_Sigma = update_Sigma_part.inverse();
    
    VectorXd new_res_y = y - alpha0 - (X * alpha) - (alpha_p * treat);
    
    VectorXd update_mu_part = (1.0/sigmaSq) * new_M.transpose() * new_res_y;
    VectorXd update_mu = update_Sigma * update_mu_part;
    
    VectorXd z = sampleMultivariateNormal(update_mu, update_Sigma, 1).row(0);
    int idxx;
    for(int k=0; k<update_size; k++){
      idxx = update_idx(k);
      eff_coeff(idxx, 2) = z(k);
    }
  }
  return eff_coeff;
}

// 5. delta udpate
MatrixXd update_delta(MatrixXd eff_coeff,
                      const double V_delta,
                      const VectorXd& psi,
                      const double sigmaSq,
                      const double theta_omega,
                      const VectorXd& y,
                      const VectorXd& alpha0,
                      const MatrixXd& M,
                      const VectorXd& alpha,
                      const double alpha_p,
                      const VectorXd& treat,
                      const MatrixXd& X){
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  int q = delta.rows(), g_omega = omega.sum(), move_2;
  
  if(g_omega == 0){
    move_2 = 1;
  }else if(g_omega == q){
    move_2 = 2;
  }else{
    move_2 = ran_sample(3);
  }
  
  MatrixXd result1 = MatrixXd::Zero(q, 4);
  MatrixXd result2 = MatrixXd::Zero(q, 4);
  
  switch(move_2){
  case 1: // add step
    result1 = add_step1(eff_coeff, V_delta, psi, sigmaSq, theta_omega, y, alpha0, M, alpha, alpha_p, treat, X);
    break;
  case 2: // delete step
    result1 = delete_step1(eff_coeff, V_delta, psi, sigmaSq, theta_omega, y, alpha0, M, alpha, alpha_p, treat, X);
    break;
  case 3: // swap step
    result1 = swap_step1(eff_coeff, V_delta, psi, sigmaSq, theta_omega, y, alpha0, M, alpha, alpha_p, treat, X);
    break;
  }
  eff_coeff = result1;
  
  // refining step
  result2 = refining_step1(eff_coeff, psi, sigmaSq, y, alpha0, M, alpha, alpha_p, treat, X);
  eff_coeff = result2;
  
  return eff_coeff;
}

// 5-1. delta udpate
MatrixXd update_delta2(MatrixXd eff_coeff,
                      const double V_delta,
                      const VectorXd& psi,
                      const double sigmaSq,
                      const double theta_omega,
                      const VectorXd& y,
                      const VectorXd& alpha0,
                      const MatrixXd& M,
                      const VectorXd& alpha,
                      const double alpha_p,
                      const VectorXd& treat,
                      const MatrixXd& X){
  
  VectorXd gamma = eff_coeff.col(1);
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  
  int q = delta.rows(), move_2;
  int g_gamma = gamma.sum(), g_omega = omega.sum();
  
  MatrixXd result1 = MatrixXd::Zero(q, 4);
  MatrixXd result2 = MatrixXd::Zero(q, 4);
  
  if(g_gamma == 0){
    
    result1 = eff_coeff;
    
  }else{
    
    if(g_omega == 0){
      move_2 = 1;
    }else if(g_omega == g_gamma){
      move_2 = 2;
    }else{
      move_2 = ran_sample(3);
    }
    
    switch(move_2){
    case 1: // add step
      result1 = add_step2(eff_coeff, V_delta, psi, sigmaSq, theta_omega, y, alpha0, M, alpha, alpha_p, treat, X);
      break;
    case 2: // delete step
      result1 = delete_step2(eff_coeff, V_delta, psi, sigmaSq, theta_omega, y, alpha0, M, alpha, alpha_p, treat, X);
      break;
    case 3: // swap step
      result1 = swap_step2(eff_coeff, V_delta, psi, sigmaSq, theta_omega, y, alpha0, M, alpha, alpha_p, treat, X);
      break;
    }
  }
  
  eff_coeff = result1;
  
  // refining step
  result2 = refining_step1(eff_coeff, psi, sigmaSq, y, alpha0, M, alpha, alpha_p, treat, X);
  eff_coeff = result2;
  
  return eff_coeff;
}


#endif
