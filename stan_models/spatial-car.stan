data {
    int<lower=0> N;    // the number of observations
    vector[N] y;       // the obersved values
    matrix[N, N] A;    // the N times N adjacency matrix 
    matrix[N, N] D;    // the N times N diagonal matrix of neighbors
}
parameters {
    vector[N] eta;    //
    real mu;
    real<lower=0> sigma2;
    real<lower=0> tau2;
    real<lower=0, upper=1> phi;
}
// transformed parameters {
//     vector[N] y_pred = mu + w_s;
// }
model {
    matrix[N, N] Sigma_inv = (D - phi * A) / tau2;
    eta ~ multi_normal_prec(rep_vector(0, N), Sigma_inv);
    mu ~ normal(0, 1);
    sigma2 ~ cauchy(0, 5);
    tau2 ~ cauchy(0, 5);
    y ~ normal(mu + eta, sigma2);
}
