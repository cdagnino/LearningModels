import numpy as np
import dill
import pandas as pd
from scipy import optimize as opt
import time
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append('../')
import src

#GMM parameters
maxiters = 50 #120. About 2 minutes per iteration
time_periods = 40 #Maximum spell_t to consider
min_periods = 3 #Min window period for standard deviation

#Parameter limits that make sense for the product (Hand-picked this time)
optimization_limits = [(-4, 0.05), (-5, 4), (1.35, 0.2), (-1, 1)]

#Load policy and value function
#####################
#file_n = "2018-10-5vfi_dict.dill" #VF for simulsted data
file_n = "2019-4-12medium_prod_vfi_dict.dill"
#file_n = "" #Work Macbook
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


#dataframe and standard deviation
cleaned_data = "../../firm_learning/data/cleaned_data/"

df = pd.read_csv(cleaned_data + "medium_prod_for_gmm.csv")
std_devs = (df.groupby('firm').level_prices.rolling(window=4, min=3)
            .std().reset_index()
            .rename(columns={'level_1': 't', 'level_prices': 'std_dev_prices'}))

df = pd.merge(df, std_devs, on=['firm', 't'], how='left')

mean_std_observed_prices = df.groupby('t').std_dev_prices.mean()[min_periods:]

#Mix Max scaling for xs
xs = df.groupby('firm').xs.first().values
scaler = MinMaxScaler()
xs = scaler.fit_transform(xs.reshape(-1, 1)).flatten()

Nfirms = len(xs)
# Just add zeroes. Makes sense for the gmm estimation
prior_shocks = src.gen_prior_shocks(Nfirms, σerror=0)


# Optimization
######################
def error_w_data(θ) -> float:
    return src.gmm_error(θ, policyF, xs,
                         mean_std_observed_prices=mean_std_observed_prices, df=df,
                         prior_shocks=prior_shocks, min_periods=min_periods)


start = time.time()
optimi = opt.differential_evolution(error_w_data, optimization_limits,
                                    maxiter=maxiters)

# Print results
#################

time_taken = time.time()/60 - start/60
print("Taken {0} minutes for {1} iterations. {2} per iteration".format(
      time_taken, maxiters, time_taken/maxiters))

print("Success: ", optimi.success)
print("Optimizados: ", np.round(optimi.x, 2))
