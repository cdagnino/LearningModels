import numpy as np
import dill
import pandas as pd
from scipy import optimize as opt
import time
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append('../')
import src
from telepyth import TelepythClient
import os
tp = TelepythClient(token=os.environ['telepyth_token'])

np.random.seed(383461)


#GMM parameters
################
maxiters = 100 #100, 8.6 minutos por iteración para differential_evolution
time_periods = 40 #Maximum spell_t to consider
max_t_to_consider = 33
min_periods = 3 #Min window period for standard deviation
use_logs_for_x = False
simul_repetitions = 5 #simulation repetitions
method = "mixed"  #"differential evolution", "Nelder-Mead", "mixed"

# Mixed mthod params
diff_evol_iterations = 10 #15
nelder_mead_iters = 100 #100
n_of_nelder_mead_tries= 8 #15


print(f"""Started at {time.asctime()}. Discount: {src.const.δ}.
          Method {method} with {maxiters} maxiters. Logs for x? {use_logs_for_x}""")


#Load policy and value function
#####################

#file_n = "2019-4-12medium_prod_vfi_dict.dill"
file_n = "2019-4-27medium_prod_vfi_dict.dill" #discount 0.95 (I think)
#file_n = "2019-4-28medium_prod_vfi_dict.dill" #discount 0.99 (I think)
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
std_devs = (df.groupby('firm').level_prices.rolling(window=4, min=min_periods)
            .std().reset_index()
            .rename(columns={'level_1': 't', 'level_prices': 'std_dev_prices'}))

df = pd.merge(df, std_devs, on=['firm', 't'], how='left')
df["dmd_shocks"] = np.random.normal(loc=0, scale=src.const.σ_ɛ, size=len(df))

n_firms = df.firm.nunique()
max_t_periods_in_data = df.groupby('firm').log_dmd.count().max()


mean_std_observed_prices = df.groupby('t').rolling_std_upc.mean()[min_periods:]

#Mix Max scaling for xs
if use_logs_for_x:
    xs = np.log(df.groupby('firm').xs.first().values + 0.1)
else:
    xs = (df.groupby('firm').xs.first().values + 0.1)
scaler = MinMaxScaler()
xs = scaler.fit_transform(xs.reshape(-1, 1)).flatten()

Nfirms = len(xs)


#Draw shocks
#################
prior_shocks = src.gen_prior_shocks(Nfirms, σerror=0) # Add zeroes for the gmm estimation
taste_std_normal_shocks = np.random.normal(loc=0, scale=1, size=(max_t_periods_in_data, n_firms))
b0_std_normal_shocks = np.random.normal(loc=0, scale=1, size=n_firms)


# Optimization
##############################

all_firms = df.firm.unique()
firm_lengths_ = np.empty(len(all_firms) + 1, dtype=int)
firm_lengths_[0] = 0
for i, firm_i in enumerate(all_firms):
    firm_lengths_[i + 1] = df[df.firm == firm_i].index[-1] + 1


def error_w_data(θandΞres) -> float:
    #θandΞ = np.append(θandΞres, [0.8, 0.3, 0.7])
    #Only index 0 and 2  full_optθ = [-4.   -3.38  1.1  -0.82]
    #θandΞ = np.array([θandΞres[0], 0., θandΞres[1], 0., 0.8, 0.3, 0.7])
    θandΞ = np.array([θandΞres[0], 0., θandΞres[1], 0., 0.8, θandΞres[2], θandΞres[3]])
    return src.full_gmm_error(θandΞ, policyF, xs, mean_std_observed_prices=mean_std_observed_prices,
                              prior_shocks=prior_shocks, df=df, len_df=len(df), firm_lengths=firm_lengths_,
                              simul_repetitions=simul_repetitions, taste_std_normal_shocks=taste_std_normal_shocks,
                              b0_std_normal_shocks=b0_std_normal_shocks,
                              n_firms=n_firms, max_t_to_consider=max_t_to_consider,
                              min_periods=min_periods, w=None)





start = time.time()
if method is "differential evolution":
    # Combined parameter: θandΞ. Product and demand side
    #optimization_limits_θ = [(-4, 0.05), (-5, 4), (0.2, 1.35), (-1, 1)]
    optimization_limits_θ = [(-5, 1), (0.2, 1.35)]
    # optimization_limits_Ξ = [(0.5, 0.9), (0.1, 0.5), (0.3, 0.8)]
    optimization_limits_Ξ = [] #[(0.5, 0.9), (0.1, 0.5)]
    optimization_limits = optimization_limits_θ + optimization_limits_Ξ

    optimi = opt.differential_evolution(error_w_data, optimization_limits,
                                        maxiter=maxiters, disp=True)
    final_success = optimi.sucess
    best_f = optimi.fun
    best_x = optimi.x
elif method is "Nelder-Mead":
    #x0 = np.array([-3.95, -3.62,  1.02,  0.28, 0.7, 0.3])
    #x0 = np.array([-2., -3.3, 1.1, 0.25, 0.65, 0.21])
    #x0 = np.array([-4.5, 0.8])
    x0 = np.array([-4.7, 1.1])
    optimi = opt.minimize(error_w_data, x0,  method='Nelder-Mead',
                          options={'maxiter': maxiters, 'disp': True})
    final_success = optimi.sucess
    best_f = optimi.fun
    best_x = optimi.x
elif method is "mixed":
    optimization_limits = [(-5., 1.), (0.2, 1.35), (0.05, 0.5), (0.2, 0.8)]
    final_success, f_and_x = src.mixed_optimization(error_w_data, optimization_limits,
                                                    diff_evol_iterations=diff_evol_iterations,
                                                    nelder_mead_iters=nelder_mead_iters,
                                                    n_of_nelder_mead_tries=n_of_nelder_mead_tries,
                                                    disp=True)

    winning_one = np.argmin(f_and_x[:, 0])
    best_f = f_and_x[winning_one][0]
    best_x = f_and_x[winning_one][1:]
else:
    raise ValueError(f"Method {method} isn't yet implemented")

# Print results
#################

time_taken = time.time()/60 - start/60
print("Taken {0} minutes for {1} iterations. {2} per iteration".format(
      time_taken, maxiters, time_taken/maxiters))

print("Success: ", final_success)
print("Optimizados: ", np.round(best_x, 2))
print("Function value: ", best_f)

tp.send_text(f"""
procedimiento terminado. A revisar!
Success: {final_success}
Optimizados: {np.round(best_x, 2)}
Function Value: {best_f}
Time taken (in minutes): {time_taken}
""")

