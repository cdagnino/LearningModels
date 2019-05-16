import numpy as np
import dill
import pandas as pd
from scipy import optimize as opt
import time
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append('../')
import src
from numba import njit

np.random.seed(383461)
#GMM parameters
maxiters = 2 #100, 1.2 minutos por iteración
time_periods = 40 #Maximum spell_t to consider
min_periods = 3 #Min window period for standard deviation
max_t_to_consider = 37
use_logs_for_x = False
print(f"Started at {time.asctime()}. Discount: {src.const.δ}. {maxiters} maxiters. Logs for x? {use_logs_for_x}")


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


@njit()
def generate_betas_inertia_Ξ(γ: int, taste_shocks_: np.array, b0_: np.array,
                             firm_periods: int, i_firm: int) -> np.array:
    """
    Generates the vector of beta demands for a firm for a total of t periods
    given by the parameter firm_periods

    :param firm_periods:
    :param i_firm:
    :return:
    """
    betas = np.empty(firm_periods)
    betas[0] = b0_[i_firm]
    old_beta = b0_[i_firm]
    for t_ in range(1, firm_periods):
        new_beta = src.nb_clip(γ * old_beta + taste_shocks_[t_, i_firm], -np.inf, -1.05)
        betas[t_] = new_beta
        old_beta = new_beta

    return betas


def param_array_to_dmd_constants(Ξ):
    return {'γ': Ξ[0],  'beta_shock_std': Ξ[1], 'taste_shock_std': Ξ[2]}


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


#Combined parameter: θandΞ. Product and demand side
optimization_limits_θ = [(-4, 0.05), (-5, 4), (1.35, 0.2), (-1, 1)]
optimization_limits_Ξ = [(0.5, 0.9), (0.1, 0.5), (0.3, 0.8)]
optimization_limits = optimization_limits_θ + optimization_limits_Ξ

simul_repetitions = 15 #simulation repetitions

def full_gmm_error(θandΞ: np.array, policyF: object, xs: np.array, mean_std_observed_prices: pd.Series,
              prior_shocks: np.array, df: pd.DataFrame, min_periods: int = 3, w=None) -> float:
    """
    Computes the gmm error of the different between the observed moments and
    the moments predicted by the model + θ

    Moments: average (over firms) standard deviation for each time period

    xs: characteristics of firms
    mean_std_observed_prices: mean (over firms) of standard deviation per t
    w: weighting matrix for GMM objective
    """
    np.random.seed(383461)

    θ = θandΞ[:4]
    Ξ = θandΞ[4::]

    lambdas0 = src.from_theta_to_lambda_for_all_firms(θ, xs, prior_shocks)

    dmd_const_dict = param_array_to_dmd_constants(Ξ)
    γ, beta_shock_std = dmd_const_dict['γ'], dmd_const_dict['beta_shock_std']
    taste_shock_std = dmd_const_dict['taste_shock_std']

    # Redo taste_shocks and b0
    taste_shocks_ = taste_std_normal_shocks*taste_shock_std
    b0_ = np.clip(src.const.mature_beta + beta_shock_std*b0_std_normal_shocks, -np.inf, -1.05)

    df["betas_inertia"] = 0.

    exp_prices = []
    #TODO: NOW definitely numba this loop!!!
    for m in range(simul_repetitions):
        for i_firm, firm in enumerate(df.firm.unique()):
            mask: pd.Series = (df.firm == firm)
            t = mask.sum()
            df.loc[mask, "betas_inertia"] = generate_betas_inertia_Ξ(γ, taste_shocks_,
                                                                     b0_, t, i_firm)

        mean_std_observed_prices_clean, mean_std_expected_prices = src.get_intersection_of_observed_and_expected_prices(
                                    mean_std_observed_prices, df, policyF, lambdas0, min_periods)
        exp_prices.append(mean_std_expected_prices)

    try:
        assert len(mean_std_observed_prices_clean) == len(mean_std_expected_prices)
    except AssertionError as e:
        e.args += (len(mean_std_observed_prices_clean), len(mean_std_expected_prices))
        raise
    exp_prices_df = pd.concat(exp_prices, axis=1)
    mean_std_expected_prices = exp_prices_df.mean(axis=1)

    max_t = max_t_to_consider
    mean_std_observed_prices_clean = mean_std_observed_prices_clean.values[:max_t]
    mean_std_expected_prices = mean_std_expected_prices.values[:max_t]

    t = len(mean_std_expected_prices)
    assert t > 0
    if w is None:
        w = np.identity(t)

    g = (1 / t) * (mean_std_expected_prices - mean_std_observed_prices_clean)
    return g.T @ w @ g




# Optimization
######################
def error_w_data(θandΞ) -> float:
    return full_gmm_error(θandΞ, policyF, xs, mean_std_observed_prices=mean_std_observed_prices,
                          prior_shocks=prior_shocks, df=df, min_periods=min_periods, w=None)


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
