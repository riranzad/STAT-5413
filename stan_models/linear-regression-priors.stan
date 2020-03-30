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

// The input data is a vector 'y' of length 'n' and an 
// 'n' times 'p' matrix of covariates (including the intercept)
data {
  int<lower=0> n;
  int<lower=0> p;
  vector[n] y;
  matrix[n, p] X;
  
}

// The parameters accepted by the model. Our model
// accepts two parameters 'beta' and 'sigma'.
parameters {
  vector[p] beta;
  real<lower=0> sigma;
  vector[n] y_rep;
}

// The transformed parameters X %*% beta for the model. 
transformed parameters {
  vector[n] mu;
  mu = X * beta;
}
// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.

model {
  y ~ normal(mu, sigma);
  beta ~ normal(0, 5);
  sigma ~ cauchy(0, 5);
  // Note that this is vectorized and equivalent to
  // for (i in 1:n) {
  //   y[i] ~ normal(mu[i], sigma);
  // }
  
  // posterior predictive distribution
  y_rep ~ normal(mu, sigma);
}
// Note: the stan file must end in a blank (new) line
