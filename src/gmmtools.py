import numpy as np
from numba import njit
import pandas as pd
import src
from scipy import optimize
from scipy import optimize as opt
from scipy.stats import truncnorm
from .from_parameters_to_lambdas import reparam_lambdas, h_and_exp_betas_eqns, jac, input_heb_to_lambda_con_norm
from typing import Tuple, List

def gen_prior_shocks(nfirms, σerror=0.005):
    return np.random.normal(loc=0., scale=σerror, size=nfirms)

@njit()
def nb_clip(x, a, b):
    """
    Clip x between a and b
    """
    if x < a:
        return a
    if x > b:
        return b
    return x


def jac_(x):
    return jac(x, βs=src.betas_transition)


#@njit()
#TODO: try to njit this
#OR: precompute lambdas values for all relevant H and EB values
def orig_from_theta_to_lambda0(x, θ, prior_shock: float, starting_values=np.array([0.1, 0.5, 0.4])):
    """
    Generates a lambda0 vector from the theta vector and x
    It passes through the entropy and expected value of betas (H, EB)

    θ = [θ10, θ11, θ20, θ21]
    x : characteristics of firms
    prior_shock: puts randomness in the relationship between theta and lambda
    """
    #TODO: bound H between 0 and log(cardinality(lambdas)) or use standardized H
    H = np.e**((θ[0] + θ[1]*x + prior_shock))
    Eβ = -np.e**(θ[2] + θ[3]*x + prior_shock) #Bound it?

    def fun_(lambda_try):
        return h_and_exp_betas_eqns(lambda_try, src.betas_transition, Eβ, H)

    #Numerical procedure to get lambda vector from H, Eβ
    #sol = optimize.root(fun_, logit(starting_values), jac=jac_)
    sol = optimize.minimize(fun_, x0=src.logit(starting_values), method='Powell')
    lambdas_sol = reparam_lambdas(sol.x)
    if not sol.success:
        # Use Nelder-Mead from different starting_value
        sol = optimize.minimize(fun_, x0=src.logit(np.array([0.6, 0.1, 0.3])), method='Nelder-Mead')
        lambdas_sol = reparam_lambdas(sol.x)
        if not sol.success:
            print(f"Theta to lambda0 didn't converge", sol.x, lambdas_sol)

    return lambdas_sol


def orig_from_theta_to_lambda_for_all_firms(θ, xs, prior_shocks):
    nfirms = len(xs)
    lambdas0 = np.empty((nfirms, 3))
    for firm_i in range(nfirms):
        lambdas0[firm_i, :] = src.from_theta_to_lambda0(xs[firm_i], θ,
                                                       prior_shocks[firm_i])

    return lambdas0


def from_theta_to_lambda0(x, θ, prior_shock: float, lambda_values_matrix,
                          h_candidates, eb_candidates, h_n_digits_precision,
                          eb_n_digits_precision, h_dict, e_dict):
    """
    Calculates lambda0 based on precalculated lambda matrix

    :param x:
    :param θ:
    :param prior_shock:
    :param lambda_values_matrix:
    :param h_candidates:
    :param eb_candidates:
    :param h_n_digits_precision:
    :param eb_n_digits_precision:
    :param h_dict:
    :param e_dict:
    :return:
    """
    #Go between 0 and 1
    H = np.clip(np.e ** ((θ[0] + θ[1] * x + prior_shock)), 0., np.log(3)) # Normalized H: between 0 and log(3)
    Eβ = np.clip(-np.e ** (θ[2] + θ[3] * x + prior_shock), -4., 1.1)
    input_point = np.array([H, Eβ])

    lambdas_sol = src.input_heb_to_lambda_con_norm(input_point, lambda_values_matrix,
                                                   h_candidates, eb_candidates,
                                                   h_n_digits_precision, eb_n_digits_precision,
                                                   h_dict, e_dict)

    return lambdas_sol


def from_theta_to_lambda_for_all_firms(θ, xs, prior_shocks,
                                       lambda_values_matrix,
                                       h_candidates, eb_candidates, h_n_digits_precision,
                                       eb_n_digits_precision, h_dict, e_dict
                                       ):
    nfirms = len(xs)
    lambdas0 = np.empty((nfirms, 3))
    for firm_i in range(nfirms):
        lambdas0[firm_i, :] = src.from_theta_to_lambda0(xs[firm_i], θ,
                                                        prior_shocks[firm_i],
                                                        lambda_values_matrix,
                                                        h_candidates, eb_candidates, h_n_digits_precision,
                                                        eb_n_digits_precision, h_dict, e_dict
                                                        )

    return lambdas0



def simulated_dmd(current_price: float, dmd_shock: float) -> float:
    """
    Generates a quantity base on a model of the real dmd function

    :param current_price: price chosen at t by policy function. LEVEL, not log
    :param dmd_shock
    :return: demand for this period
    """
    return src.const.α + src.const.mature_beta*np.log(current_price) + dmd_shock


def simulated_dmd_w_inertia(current_price: float, dmd_shock: float, beta_inertia: float) -> float:
    """
    Generates a quantity base on a model of the real dmd function. Incorporates
    demand intertia or taste shocks

    :param current_price: price chosen at t by policy function. LEVEL, not log
    :param dmd_shock:
    :return: demand for this period
    """
    return src.const.α + beta_inertia*np.log(current_price) + dmd_shock


def generate_betas_inertia(time_periods: int) -> np.ndarray:
    """
    Generate an array of beta_inertia values for t time periods
    """
    betas = np.empty(time_periods)
    taste_shocks = np.random.normal(loc=0, scale=src.const.taste_shock_std, size=time_periods)

    b0 = np.clip(np.random.normal(loc=src.const.mature_beta, scale=src.const.beta_shock_std), -np.inf, -1.05)
    betas[0] = b0
    old_beta = b0
    for t in range(1, time_periods):
        new_beta = np.clip(src.const.γ * old_beta + taste_shocks[t], -np.inf, -1.05)
        betas[t] = new_beta
        old_beta = new_beta

    return betas


@njit()
def generate_betas_inertia_Ξ(γ: int, taste_shocks_: np.array, b0_: np.array,
                             firm_periods: int, i_firm: int) -> np.array:
    """
    Generates the vector of beta demands for a firm for a total of t periods
    (given by the parameter firm_periods)
    it takes demand side parameters γ, taste_shocks_ and initials betas b0_

    :param γ: AR(1) parameter for demand
    :param taste_shocks_: matrix of taste shocks. One for each firm, time_period
    :param b0_: draws for demand elasticities at time 0
    :param firm_periods:
    :param i_firm: firm location in array
    :return: array of betas for that a firm
    """

    betas = np.empty(firm_periods)
    betas[0] = b0_[i_firm]
    old_beta = b0_[i_firm]
    for t_ in range(1, firm_periods):
        new_beta = src.nb_clip(γ * old_beta + taste_shocks_[t_, i_firm], -np.inf, -1.05)
        betas[t_] = new_beta
        old_beta = new_beta

    return betas


# TODO: speed up this function. Can't jit it because policyF is a scipy LinearNDInterpolation f
# But I could write it with explicit parameters (some sort of Interpolation?) and jit!
def generate_pricing_decisions(policyF, lambda0: np.ndarray,
                               demand_obs: np.ndarray, dmd_shocks: np.ndarray,
                               betas_inertia: np.ndarray, use_real_dmd=False,
                               use_inertia_dmd=True) -> np.ndarray:
    """
    Generates a vector of pricing for one firm based on the policy function
    (could be vectorized later!)
    """
    current_lambdas = lambda0
    level_price_decisions = np.empty_like(demand_obs)

    for t, log_dmd in enumerate(demand_obs):
        level_price = policyF(current_lambdas[:-1])
        level_price_decisions[t] = level_price

        if use_real_dmd:
            dmd_to_update_lambda = log_dmd
        else:
            if use_inertia_dmd:
                dmd_to_update_lambda = simulated_dmd_w_inertia(level_price, dmd_shocks[t],
                                                               betas_inertia[t])
            else:
                dmd_to_update_lambda = simulated_dmd(level_price, dmd_shocks[t])

        # lambda updates: log_dmd: Yes, level_price: Yes
        new_lambdas = src.update_lambdas(dmd_to_update_lambda, src.dmd_transition_fs, current_lambdas,
                                         action=level_price, old_state=1.2)
        current_lambdas = new_lambdas

    return level_price_decisions


def generate_mean_std_pricing_decisions(df, policyF, lambdas_at_0, min_periods=3,
                                        correct_std_dev=False):
    """
    Lambdas0: starting priors for each of the N firms
    """
    pricing_decision_dfs = []
    for i, firm in enumerate(df.firm.unique()):
        prices = generate_pricing_decisions(policyF, lambdas_at_0[i],
                                            df[df.firm == firm].log_dmd.values,
                                            df[df.firm == firm].dmd_shocks.values,
                                            df[df.firm == firm].betas_inertia.values)
        pricing_decision_dfs.append(pd.DataFrame({'level_prices': prices,
                                                  'firm': np.repeat(firm, len(prices))
                                                  }))

    pricing_decision_df = pd.concat(pricing_decision_dfs, axis=0)

    #TODO fill this in
    if correct_std_dev:
        pass
        ##sort by firm, upc_id, week
        #window=4, min_periods=3, ddof=0  group_vars='UPC_INT'
        #df.groupby(group_vars)[price_var]
        #  .rolling(min_periods=min_periods, window=window).std(ddof=ddof))
        #mean_std_observed_prices = df.groupby('t').rolling_std_upc.mean()[min_periods:]
    else:
        std_dev_df = (pricing_decision_df.groupby('firm').level_prices.rolling(window=4, min=min_periods)
                                         .std().reset_index()
                                         .rename(columns={'level_1': 't',
                                                          'level_prices': 'std_dev_prices'}))
        return std_dev_df.groupby('t').std_dev_prices.mean()[min_periods:]


def get_intersection_of_observed_and_expected_prices(mean_std_observed_prices: pd.Series,
                                                     df: pd.DataFrame, policyF,
                                                     lambdas0, min_periods):
    """
    Generates expected prices, eliminate nulls and finds
    intersection of observed and expected moments

    :param df:
    :param policyF:
    :param lambdas_at_0:
    :param min_periods:
    :return:
    """
    mean_std_expected_prices = generate_mean_std_pricing_decisions(df, policyF,
                                                                   lambdas0, min_periods)

    mean_std_observed_prices = mean_std_observed_prices[pd.notnull(mean_std_observed_prices)]
    mean_std_expected_prices = mean_std_expected_prices[pd.notnull(mean_std_expected_prices)]
    index_inters = np.intersect1d(mean_std_observed_prices.index,
                                  mean_std_expected_prices.index)

    mean_std_observed_prices = mean_std_observed_prices.loc[index_inters]
    mean_std_expected_prices = mean_std_expected_prices.loc[index_inters]

    return mean_std_observed_prices, mean_std_expected_prices


def prepare_df_for_estimation(df):
    pass


def gmm_error(θ: np.array, policyF: object, xs: np.array, mean_std_observed_prices: pd.Series,
              prior_shocks: np.array, df: pd.DataFrame, lambda_matrix_dict,
              min_periods: int = 3, w=None) -> float:
    """
    Computes the gmm error of the different between the observed moments and
    the moments predicted by the model + θ

    Moments: average (over firms) standard deviation for each time period

    xs: characteristics of firms
    mean_std_observed_prices: mean (over firms) of standard deviation per t
    w: weighting matrix for GMM objective
    """
    lambda_values_matrix = lambda_matrix_dict['lambda_matrix']
    h_candidates, eb_candidates = lambda_matrix_dict['h_candidates'], lambda_matrix_dict['eb_candidates']
    h_n_digits_precision = lambda_matrix_dict['h_n_digits_precision']
    eb_n_digits_precision = lambda_matrix_dict['eb_n_digits_precision']
    h_dict, e_dict = lambda_matrix_dict['h_dict'], lambda_matrix_dict['e_dict']
    lambdas0 = from_theta_to_lambda_for_all_firms(θ, xs, prior_shocks, lambda_values_matrix,
                                            h_candidates, eb_candidates, h_n_digits_precision,
                                            eb_n_digits_precision, h_dict, e_dict)


    mean_std_observed_prices, mean_std_expected_prices = (
             get_intersection_of_observed_and_expected_prices(mean_std_observed_prices,
                                                              df, policyF, lambdas0, min_periods))

    try:
        assert len(mean_std_observed_prices) == len(mean_std_expected_prices)
    except AssertionError as e:
        e.args += (len(mean_std_observed_prices), len(mean_std_expected_prices))
        raise

    t = len(mean_std_expected_prices)
    assert t > 0
    if w is None:
        w = np.identity(t)
    #g = (1 / t) * (mean_std_expected_prices - mean_std_observed_prices)[:, np.newaxis]
    #return (g.T @ w @ g)[0, 0]
    g = (1 / t) * (mean_std_expected_prices.values - mean_std_observed_prices.values)
    return g.T @ w @ g


# Full GMM: learning and demand parameters
##########################################
@njit()
def inner_loop_with_numba_unbalanced(simul_repetitions_, firm_lengths: np.array, n_firms: int, len_df: int, γ,
                                     taste_shocks, b0):
    betas_inertia_by_m = np.empty((len_df, simul_repetitions_))
    for m in range(simul_repetitions_):
        for i_firm in range(n_firms):
            betas_inertia_by_m[firm_lengths[i_firm]:firm_lengths[i_firm + 1], m] = \
                (src.generate_betas_inertia_Ξ(γ, taste_shocks, b0,
                                              firm_lengths[i_firm + 1] - firm_lengths[i_firm],
                                              i_firm))
    return betas_inertia_by_m


def param_array_to_dmd_constants(Ξ):
    return {'γ': Ξ[0],  'beta_shock_std': Ξ[1], 'taste_shock_std': Ξ[2]}


def full_gmm_error(θandΞ: np.array, policyF: object, xs: np.array, mean_std_observed_prices: pd.Series,
                   prior_shocks: np.array, df: pd.DataFrame, len_df, firm_lengths,
                   simul_repetitions: int, taste_std_normal_shocks: np.array,
                   b0_std_normal_shocks: np.array, n_firms: int, max_t_to_consider: int,
                   lambda_matrix_dict: dict,
                   min_periods: int=3, w=None) -> float:
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

    lambda_values_matrix = lambda_matrix_dict['lambda_matrix']
    h_candidates, eb_candidates = lambda_matrix_dict['h_candidates'], lambda_matrix_dict['eb_candidates']
    h_n_digits_precision = lambda_matrix_dict['h_n_digits_precision']
    eb_n_digits_precision = lambda_matrix_dict['eb_n_digits_precision']
    h_dict, e_dict = lambda_matrix_dict['h_dict'], lambda_matrix_dict['e_dict']
    lambdas0 = from_theta_to_lambda_for_all_firms(θ, xs, prior_shocks, lambda_values_matrix,
                                            h_candidates, eb_candidates, h_n_digits_precision,
                                            eb_n_digits_precision, h_dict, e_dict)

    dmd_const_dict = param_array_to_dmd_constants(Ξ)
    γ, beta_shock_std = dmd_const_dict['γ'], dmd_const_dict['beta_shock_std']
    taste_shock_std = dmd_const_dict['taste_shock_std']

    # Redo taste_shocks and b0
    taste_shocks_ = taste_std_normal_shocks*taste_shock_std
    b0_ = np.clip(src.const.mature_beta + beta_shock_std*b0_std_normal_shocks, -np.inf, -1.05)


    exp_prices = []
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
    #gc.collect()
    return g.T @ w @ g



def mixed_optimization(error_f, optimization_limits: List[Tuple[float, float]], diff_evol_iterations=15,
                       nelder_mead_iters=30, n_of_nelder_mead_tries=10, disp=True):
    """
    Starts with differential evolution and then does Nelder-Mead
    :param optimization_limits:
    :return:
    """
    # Run differential evolution for a few iterations
    successes = []
    f_and_x = np.empty((n_of_nelder_mead_tries + 2, len(optimization_limits) + 1))
    # Run differential evolution for a few iterations
    diff_evol_opti = opt.differential_evolution(error_f, optimization_limits,
                                                maxiter=diff_evol_iterations, disp=disp)
    successes.append(diff_evol_opti.success)
    f_and_x[0, :] = np.array([diff_evol_opti.fun] + list(diff_evol_opti.x))

    # One Nelder-Mead from diff_evol end
    optimi = opt.minimize(error_f, x0=diff_evol_opti.x, method='Nelder-Mead',
                          options={'maxiter': nelder_mead_iters, 'disp': disp})
    successes.append(optimi.success)
    f_and_x[1, :] = np.array([optimi.fun] + list(optimi.x))

    # TODO parallelize Nelder-Mead
    # K random points
    k_random_points = np.empty((n_of_nelder_mead_tries, len(optimization_limits)))
    for x_arg_n in range(len(optimization_limits)):
        this_opti_limits = optimization_limits[x_arg_n]
        min_, max_ = this_opti_limits[0], this_opti_limits[1]
        loc = (max_ - min_) / 2
        scale = (max_ - min_) / 4
        k_random_points[:, x_arg_n] = truncnorm.rvs(min_, max_, loc=loc, scale=scale, size=n_of_nelder_mead_tries)

    for nelder_try in range(n_of_nelder_mead_tries):
        print(f"Doing try Nelder try {nelder_try} of {n_of_nelder_mead_tries}")
        optimi = opt.minimize(error_f, k_random_points[nelder_try, :], method='Nelder-Mead',
                              options={'maxiter': nelder_mead_iters, 'disp': disp})
        successes.append(optimi.success)
        f_and_x[nelder_try + 2, :] = np.array([optimi.fun] + list(optimi.x))

    final_success = max(successes)
    return final_success, f_and_x

