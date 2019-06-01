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
import gc
from numba import njit
tp = TelepythClient(token=os.environ['telepyth_token'])

np.random.seed(383461)
#GMM parameters
maxiters = 100 #100, 8.6 minutos por iteración para differential_evolution
time_periods = 40 #Maximum spell_t to consider
max_t_to_consider = 37
min_periods = 3 #Min window period for standard deviation
use_logs_for_x = False
simul_repetitions = 5 #simulation repetitions
method = "mixed"  #"differential evolution", "Nelder-Mead", "mixed"
print(f"""Started at {time.asctime()}. Discount: {src.const.δ}.
          Method {method} with {maxiters} maxiters. Logs for x? {use_logs_for_x}""")


##########################################################
#
#
# ROUGH SKETCH OF HOW THIS SHOULD LOOK. NOT WORKING CODE
#
#
###########################################################


#Load policy and value function
#####################

prod_dict = {"Pepito": "many things here", "Rosita": "tons of other things here too"}

for prod_name, prod_config in prod_dict.items():

    #1. Load VFfile
    #2. Parameter limits: can't do them by product if I'm estimating just one set of parameters
    #                     for all products
    #3. Load data

    # 1. Load VFfile
    ################
    file_n = "2019-4-27medium_prod_vfi_dict.dill"

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

    # 2. Parameter limits that make sense for the product
    ################################################################
    optimization_limits = [(-4, 0.05), (-5, 4), (1.35, 0.2), (-1, 1)]

    # 3. Load data
    #########################
    cleaned_data = "../../firm_learning/data/cleaned_data/"

    df = pd.read_csv(cleaned_data + "medium_prod_for_gmm.csv")
    std_devs = (df.groupby('firm').level_prices.rolling(window=4, min=min_periods)
                .std().reset_index()
                .rename(columns={'level_1': 't', 'level_prices': 'std_dev_prices'}))

    df = pd.merge(df, std_devs, on=['firm', 't'], how='left')
    df["dmd_shocks"] = np.random.normal(loc=0, scale=src.const.σ_ɛ, size=len(df))

    #Fix beta_0 and taste shocks for all t and all firms
    n_firms = df.firm.nunique()
    max_t_periods_in_data = df.groupby('firm').log_dmd.count().max()
    taste_shocks = np.random.normal(loc=0, scale=src.const.taste_shock_std,
                                    size=(max_t_periods_in_data, n_firms))
    b0 = np.clip(np.random.normal(loc=src.const.mature_beta, scale=src.const.beta_shock_std, size=n_firms),
                 -np.inf, -1.05)


    @njit()
    def new_generate_betas_inertia(firm_periods: int, i_firm: int) -> np.array:
        """
        Generates the vector of beta demands for a firm for a total of t periods
        given by the parameter firm_periods

        :param firm_periods:
        :param i_firm:
        :return:
        """
        betas = np.empty(firm_periods)
        betas[0] = b0[i_firm]
        old_beta = b0[i_firm]
        for t_ in range(1, firm_periods):
            new_beta = src.nb_clip(src.const.γ * old_beta + taste_shocks[t_, i_firm], -np.inf, -1.05)
            betas[t_] = new_beta
            old_beta = new_beta

        return betas


    df["betas_inertia"] = 0.


    #New Procedure
    for i_firm, firm in enumerate(df.firm.unique()):
        mask: pd.Series = (df.firm == firm)
        t = mask.sum()
        df.loc[mask, "betas_inertia"] = new_generate_betas_inertia(t, i_firm)


    #mean_std_observed_prices = df.groupby('t').std_dev_prices.mean()[min_periods:]
    mean_std_observed_prices = df.groupby('t').rolling_std_upc.mean()[min_periods:]

    #Mix Max scaling for xs
    if use_logs_for_x:
        xs = np.log(df.groupby('firm').xs.first().values + 0.1)
    else:
        xs = (df.groupby('firm').xs.first().values + 0.1)
    scaler = MinMaxScaler()
    xs = scaler.fit_transform(xs.reshape(-1, 1)).flatten()

    Nfirms = len(xs)
    # Just add zeroes. Makes sense for the gmm estimation
    prior_shocks = src.gen_prior_shocks(Nfirms, σerror=0)

# Concat dfs. Also concat prior_shocks
# Optimization: all products together
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
