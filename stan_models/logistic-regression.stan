data {
  int<lower=0> n;
  int<lower=0> p;
  int<lower=0,upper=1> y[n];
  matrix[n, p] X;
}

parameters {
  vector[p] beta;
}

model {
  y ~ bernoulli_logit(X * beta);
  beta ~ normal(0, 1);
}

generated quantities {
  int y_rep[n];
  y_rep = bernoulli_logit_rng(X * beta);
}
// Note: the stan file must end in a blank (new) line
