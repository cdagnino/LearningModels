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
simul_repetitions = 10 #simulation repetitions
method = "differential evolution" #"differential evolution", "Nelder-Mead"
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


@njit()
def inner_loop_with_numba_unbalanced(simul_repetitions_, firm_lengths: np.array, n_firms: int, len_df: int, γ,
                                     taste_shocks, b0):
    betas_inertia_by_m = np.empty((len_df, simul_repetitions_))
    for m in range(simul_repetitions):
        for i_firm in range(n_firms):
            betas_inertia_by_m[firm_lengths[i_firm]:firm_lengths[i_firm + 1], m] = \
                (src.generate_betas_inertia_Ξ(γ, taste_shocks, b0,
                                              firm_lengths[i_firm + 1] - firm_lengths[i_firm],
                                              i_firm))
    return betas_inertia_by_m


#Draw shocks
#################
prior_shocks = src.gen_prior_shocks(Nfirms, σerror=0) # Add zeroes for the gmm estimation
taste_std_normal_shocks = np.random.normal(loc=0, scale=1, size=(max_t_periods_in_data, n_firms))
b0_std_normal_shocks = np.random.normal(loc=0, scale=1, size=n_firms)



def full_gmm_error(θandΞ: np.array, policyF: object, xs: np.array, mean_std_observed_prices: pd.Series,
                   prior_shocks: np.array, df: pd.DataFrame, len_df, firm_lengths,
                   min_periods: int = 3, w=None) -> float:
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



    exp_prices = []
    #TODO: NOW definitely numba this loop!!!
    #df["betas_inertia"] = 0.
    #for m in range(simul_repetitions):
    #    for i_firm, firm in enumerate(df.firm.unique()):
    #        mask: pd.Series = (df.firm == firm)
    #        t = mask.sum()
    #        df.loc[mask, "betas_inertia"] = src.generate_betas_inertia_Ξ(γ, taste_shocks_,
    #                                                                 b0_, t, i_firm)
    m_betas_inertia = inner_loop_with_numba_unbalanced(simul_repetitions, firm_lengths,
                                                      n_firms, len_df, γ,
                                                      taste_shocks_, b0_)

    for m in range(simul_repetitions):
        df['betas_inertia'] = m_betas_inertia[:, m]
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
    del df, exp_prices_df
    #del mean_std_expected_prices, mean_std_observed_prices_clean
    gc.collect()
    return g.T @ w @ g


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
    θandΞ = np.array([θandΞres[0], 0., θandΞres[1], 0., 0.8, 0.3, 0.7])
    return full_gmm_error(θandΞ, policyF, xs, mean_std_observed_prices=mean_std_observed_prices,
                          prior_shocks=prior_shocks, df=df, len_df=len(df), firm_lengths=firm_lengths_,
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
elif method is "Nelder-Mead":
    #x0 = np.array([-3.95, -3.62,  1.02,  0.28, 0.7, 0.3])
    x0 = np.array([-2., -3.3, 1.1, 0.25, 0.65, 0.21])
    optimi = opt.minimize(error_w_data, x0,  method='Nelder-Mead',
                          options={'maxiter': maxiters, 'disp': True})
else:
    print(f"Method {method} isn't yet implemented")

# Print results
#################

time_taken = time.time()/60 - start/60
print("Taken {0} minutes for {1} iterations. {2} per iteration".format(
      time_taken, maxiters, time_taken/maxiters))

print("Success: ", optimi.success)
print("Optimizados: ", np.round(optimi.x, 2))

tp.send_text("procedimiento terminado. A revisar!")
