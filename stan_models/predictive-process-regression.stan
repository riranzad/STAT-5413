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
// an 'n' times '2' matrix 'coords' of spatial locations, and 
// an 'n_knots' times '2' matrix of 'knots' for the predictive process
data {
  int<lower=0> n;
  int<lower=0> p;
  int<lower=0> n_knots;
  vector[n] y;
  matrix[n, p] X;
  matrix[n, 2] coords;
  matrix[n_knots, 2] knots;
}

transformed data {
  matrix[n, n_knots] D;
  matrix[n_knots, n_knots] D_star;
  for (i in 1:n) {
    for (j in 1:n_knots) {
      D[i, j] = sqrt(sum(square(coords[i, ] - knots[j, ])));
    }
  }
  for (i in 1:n_knots) {
    for (j in 1:(i-1)) {
      D_star[i, j] = sqrt(sum(square(knots[i, ] - knots[j, ])));
      D_star[j, i] = D_star[i, j];
    }
    D_star[i, i] = 0.0;
  }
}

// The parameters accepted by the model. Our model
// accepts two parameters 'beta' and 'sigma'.
parameters {
  vector[p] beta;
  real<lower=0> sigma2;
  real<lower=0> tau2;
  real<lower=0> phi;
  vector[n_knots] eta_centered;
}

// The transformed parameters X %*% beta for the model, 
// Sigma for the covariance matrix, and L_Sigma for the
// Cholesky decomposition of the covariance matrix
transformed parameters {
  vector[n] mu;
  vector[n_knots] eta_star;
  vector[n] eta;
  cov_matrix[n_knots] Sigma_star;
  cov_matrix[n_knots] Sigma_star_inv;
  matrix[n, n_knots] c;
  
  // fixed effects
  mu = X * beta;
  
  // latent GP covariance matrix at the knots
  // for (i in 1:n_knots) {
  //   for (j in 1:(i-1)) {
  //     Sigma_star[i, j] = tau2 * exp(- D_star[i, j] / phi);
  //     Sigma_star[j, i] = Sigma_star[i, j];
  //   }
  // }
  Sigma_star = tau2 * exp(-D_star / phi);
  Sigma_star_inv = inverse(Sigma_star);

  // predictive process interpolation matrix
  c = tau2 * exp(- D / phi);
  eta_star = cholesky_decompose(Sigma_star) * eta_centered;
  eta = c * Sigma_star_inv * eta_star;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.

model {
    // priors
  phi ~ normal(0, 10);
  tau2 ~ normal(0, 5);
  sigma2 ~ normal(0, 5);
  beta ~ normal(0, 5);
  eta_centered ~ normal(0, 1);
  y ~ normal(mu + eta, sqrt(sigma2));
  // Note that this is vectorized and equivalent to
  // for (i in 1:n) {
  //   y[i] ~ normal(mu[i], sigma);
  // }
}
// Note: the stan file must end in a blank (new) line
