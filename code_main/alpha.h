#ifndef __ALPHA_H__
#define __ALPHA_H__

#include <Rcpp.h>
#include <RcppEigen.h>
#include "basic.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

//[[Rcpp::depends(RcppEigen)]]

// Library Functions
VectorXd update_alpha0(const VectorXd& y, const MatrixXd& M, const VectorXd& delta, const MatrixXd& X, const VectorXd& alpha, double alpha_p, const VectorXd& treat, double sigmaSq, double s0);

VectorXd update_alpha(const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& delta, const MatrixXd& X, double alpha_p, const VectorXd& treat, double sigmaSq, double t0);

double update_alpha_p(const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& delta, const MatrixXd& X, const VectorXd& alpha, const VectorXd& treat, double sigmaSq, double k0);

// Functions Define
// 1. alpha0 update
VectorXd update_alpha0(const VectorXd& y,
                       const MatrixXd& M,
                       const VectorXd& delta,
                       const MatrixXd& X,
                       const VectorXd& alpha,
                       double alpha_p,
                       const VectorXd& treat,
                       double sigmaSq,
                       double s0) {
    int n = y.rows();
    Rcpp::NumericVector alpha0_vec_numericv(n);
    VectorXd res_y = y - M * delta - X * alpha - alpha_p * treat;

    double update_sigma = 1.0 / ((n * 1.0) / sigmaSq + 1.0 / s0);
    double update_mu = (res_y.sum() / sigmaSq) * update_sigma;

    alpha0_vec_numericv = Rcpp::rnorm(n, update_mu, sqrt(update_sigma));
    Eigen::Map<Eigen::VectorXd> alpha0_vec(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(alpha0_vec_numericv));

    return alpha0_vec;
}

// 2. alpha update
VectorXd update_alpha(const VectorXd& y,
                      const VectorXd& alpha0,
                      const MatrixXd& M,
                      const VectorXd& delta,
                      const MatrixXd& X,
                      double alpha_p,
                      const VectorXd& treat,
                      double sigmaSq,
                      double t0) {
    int p = X.cols();

    // Calculate residuals
    Eigen::VectorXd res_y = y - alpha0 - M * delta - alpha_p * treat;

    // Compute sum of X'X and X'y
    Eigen::MatrixXd sum_x = X.transpose() * X;
    Eigen::VectorXd sum_x_res_y = X.transpose() * res_y;

    // Update sigma and mu
    Eigen::MatrixXd update_sigma_part = sum_x / sigmaSq + (1.0 / t0) * Eigen::MatrixXd::Identity(p, p);
    Eigen::LLT<Eigen::MatrixXd> lltOfA(update_sigma_part);
    // if (lltOfA.info() != Eigen::Success) {
    //     Rcpp::stop("Cholesky decomposition failed!");
    // }
    Eigen::MatrixXd update_sigma = lltOfA.solve(Eigen::MatrixXd::Identity(p, p));
    Eigen::VectorXd update_mu = update_sigma * (sum_x_res_y / sigmaSq);

    Eigen::VectorXd alpha_vec = sampleMultivariateNormal(update_mu, update_sigma, 1).row(0);

    return alpha_vec;
}

// 3. alpha prime update
double update_alpha_p(const VectorXd& y,
                      const VectorXd& alpha0,
                      const MatrixXd& M,
                      const VectorXd& delta,
                      const MatrixXd& X,
                      const VectorXd& alpha,
                      const VectorXd& treat,
                      double sigmaSq,
                      double k0) {
    // Calculate residuals
    VectorXd res_y = y - alpha0 - M * delta - X * alpha;
    VectorXd treatSq = treat.array().square();
    
    double update_mu_part = res_y.dot(treat);
    double update_sigma = 1.0 / ((treatSq.sum() / sigmaSq) + (1.0 / k0));
    double update_mu = update_mu_part * update_sigma / sigmaSq;

    double alpha_p      = Rcpp::rnorm(1, update_mu, sqrt(update_sigma))(0);

    return alpha_p;
}

#endif