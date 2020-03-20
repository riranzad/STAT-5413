//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' = 'X beta` and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'n', an 
// 'n' times 'p' matrix of covariates (including the intercept),
// and 'coords' is an 'n' times '2' matrix of spatial locations
data {
  int<lower=0> n;
  int<lower=0> p;
  vector[n] y;
  matrix[n, p] X;
  matrix[n, 2] coords;
}

// calculate the Euclidean distance between observed locations
transformed data {
  matrix[n, n] D;
  for (i in 1:n) {
    for (j in 1:(i-1)) {
      D[i, j] = sqrt(sum(square(coords[i, ] - coords[j, ])));
      D[j, i] = D[i, j];
    }
    D[i, i] = 0.0;
  }
}

// The parameters accepted by the model. Our model
// accepts the parameters 'beta', 'sigma2', 'tau2' and 'phi'.
parameters {
  vector[p] beta;
  real<lower=0> sigma2;
  real<lower=0> tau2;
  real<lower=0> phi;
}

// The transformed parameters X %*% beta for the model, 
// Sigma for the covariance matrix, and L_Sigma for the
// Cholesky decomposition of the covariance matrix
transformed parameters {
  vector[n] mu;
  cov_matrix[n] Sigma;
  matrix[n, n] L_Sigma;
  
  mu = X * beta;
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      Sigma[i, j] = tau2 * exp(- D[i, j] / phi);
      Sigma[j, i] = Sigma[i, j];
    }
  }
  for (i in 1:n) {
    Sigma[i, i] = sigma2 + tau2;
  }
  L_Sigma = cholesky_decompose(Sigma);
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and Cholesky of covariance matrix L_Sigma.
model {
  // priors
  phi ~ normal(0, 10);
  tau2 ~ normal(0, 5);
  sigma2 ~ normal(0, 5);
  beta ~ normal(0, 5);
  // likelihood
  y ~ multi_normal_cholesky(mu, L_Sigma);
  // Note that this is vectorized and equivalent to
  // for (i in 1:n) {
  //   y[i] ~ normal(mu[i], sigma);
  // }
}
// Note: the stan file must end in a blank (new) line
