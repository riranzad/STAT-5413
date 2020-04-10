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

// The parameters accepted by the model. 
parameters {
  vector[p] beta;
  vector[n] y_rep;
}
model {
  y ~ bernoulli_logit(X * beta);
  beta ~ normal(0, 1);
  y_rep ~ bernoulli_logit(X * beta);
}
// Note: the stan file must end in a blank (new) line
