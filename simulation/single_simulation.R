library(MASS)
library(invgamma)
library(dplyr)
library(mvtnorm)
library(Rcpp)
sourceCpp("~/code_main/main.cpp")
load("~/code_main/covM_new.RData")
load("~/code_main/idx.RData")

set.seed(temp)
# Set True Value ---------------------------------------------------------------
n = 466; q = 298; p = 3

X = mvrnorm(n, mu = rep(0,p), Sigma = diag(p))
treat = rnorm(n, 0.5*X[,1]+0.2*X[,2]+0.7*X[,3], 1)

beta0_vec_t = rep(0.1,q)

tau_t = rep(0, q)
tau_t[final_idx] = c(rep(-0.12,5),rep(-0.08,5),rep(-0.04,5),rep(0.04,5),rep(0.08,5),rep(0.12,5))
B_t         = matrix(0.1, nrow = p, ncol = q)

M = matrix(0,n,q)
for(i in 1:n){
  M[i,] = mvrnorm(1, beta0_vec_t + tau_t*treat[i] + (t(B_t) %*% X[i,]), 0.5*covM)
}

alpha0_t   = 2
delta_t    = rep(0, q)
delta_t[final_idx] = c(rep(c(0.5,1,1.5,0,0),6))
alpha_t    = rep(2, p)
alpha_p_t  = 2
sigmaSq_t  = 0.5

y = rep(0,n)
for(i in 1:n){
  y[i] = rnorm(1, alpha0_t + t(delta_t)%*%M[i,] + t(alpha_t)%*%X[i,] + alpha_p_t*treat[i], sqrt(sigmaSq_t))
}

outcome = cmaVS_final2(treat=treat, M=M, X=X, y=y,
                       iter=350000, burn_in=175000, thin=175,
                       theta_gamma=-2.2, theta_omega=0.1,
                       mu_lambda=0, h_lambda=100,
                       nu_element=3, psi_element=3, 
                       h0=100, c0=100, s0=100, t0=100, k0=100,
                       V_tau=0.01, V_delta=0.01, V_lambda=0.01,
                       mu0=0, mu1=0, nu0=6, sigmaSq0=1/3, nu1=6, sigmaSq1=1/3,
                       update_prop=c(1,1,1,1,1,1,1,1,1,1), init = 0.3, eta=0.25)
