data {
  int<lower=1> N; // Number of sites
  int<lower=1> J; // Number of temporal replications
  real y[N]; // Observation
  real x[N]; 
  int<lower = 1> K; // Number of categories
  int<lower=0, upper=K> group[N]; 
  real mu[K];
  real sigma[K];
}
parameters {
  real beta0;
  real beta1;
  real eta[N];
  real theta[K];
  real<lower=0> tau;
}
model {
  // Priors
  for (k in 1:K){
    theta[k] ~ normal(mu[k], sigma[k]);
  }
  beta0 ~ normal(0, 2);
  beta1 ~ normal(0, 2);
  tau ~ normal(0, 1);
  
  // Likelihood
  for(i in 1:N){
    eta[i] ~ normal(theta[group[i]], 1);
    y[i] ~ normal(beta0 + beta1 * x[i] + eta[i], tau);
  }
}