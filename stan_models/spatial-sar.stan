data {
  int<lower=0> N;
  vector[N] y;
  matrix[N, N] A;
  matrix[N, N] D;
}
transformed data {
  matrix[N, N] I = diag_matrix(rep_vector(1, N));
  matrix[N, N] A_D_inv;
  A_D_inv = A * inverse(D);
}
parameters {
  vector[N] eta;
  real mu;
  real<lower=0> sigma2;
  real<lower=0> tau2;
  real<lower=0,upper=1> phi;
}
transformed parameters {
  vector[N] y_pred = mu + eta;
}
model {
  matrix[N,N] C = I - phi * A_D_inv;
  matrix[N,N] Sigma_inv = C * C' / tau2;
  eta ~ multi_normal_prec(rep_vector(0, N), Sigma_inv);
  mu ~ normal(0, 1);
  sigma2 ~ cauchy(0, 5);
  tau2 ~ cauchy(0, 5);
  y ~ normal(mu + eta, sigma2);
}

