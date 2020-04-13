data {
  int<lower=0> n;
  int<lower=0> p;
  int<lower=0> y[n];
  matrix[n, p] X;
}

// The parameters accepted by the model. 
parameters {
  vector[p] beta;
}

model {
  y ~ poisson_log(X * beta);
  beta ~ normal(0, 5);
}

generated quantities {
  int y_rep[n];
  y_rep = poisson_log_rng(X * beta);  
}
// Note: the stan file must end in a blank (new) line
