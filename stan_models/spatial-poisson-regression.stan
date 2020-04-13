data {
  int<lower=0> n;
  int<lower=0> p;
  int<lower=0> y[n];
  matrix[n, p] X;
  matrix[n, 2] locs;
}

transformed data {
  matrix[n, n] D;
  for (i in 1:n) {
    for (j in 1:(i-1)) {
      D[i, j] = sqrt(sum(square(locs[i, ] - locs[j, ])));
      D[j, i] = D[i, j];
    }
    D[i, i] = 0.0;
  }
}

parameters {
  vector[p] beta;
  real<lower=0> tau2;
  real<lower=0> phi;
  vector[n] eta_center;
}

transformed parameters {
  cov_matrix[n] Sigma;
  matrix[n, n] L_Sigma;
  Sigma = tau2 * exp( - D / phi);
  L_Sigma = cholesky_decompose(Sigma);
}

model {
  tau2 ~ student_t(3, 0, 1);  // student t prior with df 3, mean 0, sd 1
  phi ~ student_t(3, 0, 1);   // student t prior with df 3, mean 0, sd 1
  to_vector(eta_center) ~ std_normal();
  y ~ poisson_log(X * beta + L_Sigma * eta_center);  
  beta ~ normal(0, 5);
}

generated quantities {
  int y_rep[n];
  y_rep = poisson_log_rng(X * beta + L_Sigma * eta_center);  
}
// Note: the stan file must end in a blank (new) line
