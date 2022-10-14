// consider https://mc-stan.org/docs/stan-users-guide/multivariate-hierarchical-priors.html
// for more complexity and dependence among theta
data {
  int<lower=1> N; // Number of sites
  int<lower=1> J; // Number of visits (fixed for now, doesnt have to be)
  int<lower = 1> K; // Number of categories
  int<lower=0, upper=K> group[N];  // Indicator vector
  int<lower=0> sumy[N]; // Sum of obs for all visits
  real x[N]; 
  real m[K]; // prior means from mapping the experts
  real<lower = 0> s[K]; // prior sds from mapping the experts
}
parameters {
  real beta0;
  real beta1;
  real theta[K];
  real mu[K];
  real<lower=0, upper=1> p; // Detection probability
  real<lower=0> tau;
}
model {
  // hyper priors
  for(k in 1:K){
    mu[k] ~ normal(m[k], s[k]);
  }

  // hierarchical prior
  for (k in 1:K){
    theta[k] ~ normal(mu[k], tau);
  }
  
  // other stuff
  beta0 ~ normal(0, 2);
  beta1 ~ normal(0, 2);
    
  // Likelihood
  for (i in 1:N) {
    if(sumy[i] > 0) {
      // Occurred and observed
      1 ~ bernoulli(inv_logit(beta0 + beta1 * x[i] + theta[group[i]]));
      sumy[i] ~ binomial(J, p);
    } else {
      target += log_sum_exp(
        // Occurred and not observed
        bernoulli_lpmf(1 | inv_logit(beta0 + beta1 * x[i] + theta[group[i]])) + bernoulli_lpmf(0 | p) * J,
        // Not occurred
        bernoulli_lpmf(0 | inv_logit(beta0 + beta1 * x[i] + theta[group[i]]))
      );
    }
  }
}
generated quantities{
  real<lower = 0, upper = 1> psi[N];
  for(i in 1:N){
    psi[i] = inv_logit(beta0 + beta1 * x[i] + theta[group[i]]);
  }
}




