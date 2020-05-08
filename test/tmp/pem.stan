functions {
      int bin_search(real x, int min_val, int max_val){
    int range = (max_val - min_val + 1) / 2;
    int mid_pt = min_val + range;
    int out;
    while (range > 0) {
        if (x == mid_pt) {
            out = mid_pt;
            range = 0;
        } else {
            range = (range + 1) / 2; 
            mid_pt = x > mid_pt ? mid_pt + range: mid_pt - range; 
        }
    }
    return out;
}

    real[] sho(real t,real[] internal_var___u,real[] internal_var___p,real[] x_r,int[] x_i) {
  real internal_var___du[2];
  internal_var___du[1] = internal_var___u[1] * internal_var___p[1] - internal_var___u[1] * internal_var___u[2];
  internal_var___du[2] = -3 * internal_var___u[2] + internal_var___u[1] * internal_var___u[2];
  return internal_var___du;
}

  }
  data {
    real u0[2];
    int<lower=1> T;
    real internal_var___u[T,1];
    real t0;
    real ts[T];
  }
  transformed data {
    real x_r[0];
    int x_i[0];
  }
  parameters {
    row_vector<lower=0>[1] sigma1;
    real theta1;real theta2;
  }
  transformed parameters{
    real theta[2];
    theta[1] = theta1;theta[2] = theta2;
  }
  model{
    real u_hat[T,2];
    sigma1 ~ inv_gamma(3.0, 3.0);
    theta[1] ~normal(1.0, 0.5);theta[2] ~normal(1.5, 0.5);
    u_hat = integrate_ode_rk45(sho, append_array(theta[1:1],{1.0}), t0, ts, theta[2:2], x_r, x_i, 0.001, 1.0e-6, 100000);
    for (t in 1:T){
      internal_var___u[t,:] ~ normal(u_hat[t,1],sigma1);
      }
  }
  generated quantities{
    real u_hat[T,2];
    u_hat = integrate_ode_rk45(sho, u0, t0, ts, theta, x_r, x_i, 0.001, 1.0e-6, 100000);
  }
