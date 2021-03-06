import numpy as np
import dill
import pandas as pd
from scipy import optimize as opt
import time
import sys
sys.path.append('../')
import src

#GMM parameters
maxiters = 50 #120. About 2 minutes per iteration

# Simulation parameters \
########################
σerror= 0.005 #0.01
Nfirms = 300
time_periods = 40
min_periods= 3

#Suitable for logistic
β10, β11 = -2, 3.5
β20, β21 = 1.3, -2.
betas = [β10, β11, β20, β21]


#Load policy and value function
#####################
file_n = "2018-10-5vfi_dict.dill" #Personal Macbook
#file_n = "2019-2-16vfi_dict.dill" #Work Macbook
with open('../data/' + file_n, 'rb') as file:
    data_d = dill.load(file)

lambdas = src.generate_simplex_3dims(n_per_dim=data_d['n_of_lambdas_per_dim'])
price_grid = np.linspace(data_d['min_price'], data_d['max_price'])

policy = data_d['policy']
valueF = data_d['valueF']
lambdas_ext = src.generate_simplex_3dims(n_per_dim=
                                         data_d['n_of_lambdas_per_dim'])

#Interpolate policy (level price). valueF is already a function
policyF = src.interpolate_wguess(lambdas_ext, policy)


# Start simulation
######################
xs = np.abs(np.random.normal(0, 0.18, size=Nfirms))
prior_shocks = src.gen_prior_shocks(Nfirms, σerror=σerror)
dmd_shocks = src.generate_dmd_shocks(n=Nfirms, t=time_periods, dmd_σϵ=src.const.σ_ɛ)

#dataframe and standard deviation
df = src.simulate_all_firms(Nfirms, valueF, policyF, xs, θ=betas,
                            dmd_shocks=dmd_shocks, prior_shocks=prior_shocks)
std_devs = (df.groupby('firm').level_prices.rolling(window=4, min=3)
            .std().reset_index()
            .rename(columns={'level_1': 't',
                            'level_prices': 'std_dev_prices'}))

df = pd.merge(df, std_devs, on=['firm', 't'], how='left')

mean_std_observed_prices = df.groupby('t').std_dev_prices.mean()[min_periods:]


# Optimization
######################
def error_w_data(θ) -> float:
    return src.gmm_error(θ, policyF, xs,
                         mean_std_observed_prices=mean_std_observed_prices, df=df,
                         prior_shocks=prior_shocks, min_periods=min_periods)


start = time.time()

optimi = opt.differential_evolution(error_w_data, [(-2.5, 0.5), (2.0, 4.0),
                                                   (0.5, 2), (-3., 1.)],
                                    maxiter=maxiters)


# Print results
#################

time_taken = time.time()/60 - start/60
print("Taken {0} minutes for {1} iterations. {2} per iteration".format(
      time_taken, maxiters, time_taken/maxiters))

print("Success: ", optimi.success)
print("Valores reales: ", betas)
print("Optimizados: ", np.round(optimi.x, 2))
print("Errores porcentuales (de 1 a 100%): ",
      np.round(200*(np.array(betas) - optimi.x)/(np.array(betas) + optimi.x), 2))