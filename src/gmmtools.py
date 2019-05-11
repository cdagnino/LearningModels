import numpy as np
from numba import njit
import pandas as pd
import src
from scipy import optimize
from .from_parameters_to_lambdas import logit, force_sum_to_1, reparam_lambdas, h_and_exp_betas_eqns, jac


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
def from_theta_to_lambda0(x, θ, prior_shock: float, starting_values=np.array([0.1, 0.5])):
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
    lambdas_sol = force_sum_to_1(reparam_lambdas(sol.x))
    if not sol.success:
        # Use Nelder-Mead from different starting_value
        sol = optimize.minimize(fun_, x0=src.logit(np.array([0.1, 0.08])), method='Nelder-Mead')
        lambdas_sol = force_sum_to_1(reparam_lambdas(sol.x))
        if not sol.success:
            print(f"Theta to lambda0 didn't converge", sol.x, lambdas_sol)

    return lambdas_sol


def from_theta_to_lambda_for_all_firms(θ, xs, prior_shocks):
    nfirms = len(xs)
    lambdas0 = np.empty((nfirms, 3))
    for firm_i in range(nfirms):
        lambdas0[firm_i, :] = src.from_theta_to_lambda0(xs[firm_i], θ,
                                                       prior_shocks[firm_i])

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

# TODO: speed up this function. Can't jit it because policyF is a scipy LinearNDInterpolation f
# But I could write it with explicit parameters and jit!
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


def std_moments_error(θ: np.ndarray, policyF, xs, mean_std_observed_prices,
                      prior_shocks, df, min_periods=3) -> float:
    """
    Computes the **norm** (not gmm error) of the different between the
    observed moments and the moments predicted by the model + θ

    Moments: average (over firms) standard deviation for each time period

    x: characteristics of firms
    mean_std_observed_prices: mean (over firms) of standard deviation per t
    """
    # Generate one lambda0 per firm
    lambdas0 = from_theta_to_lambda_for_all_firms(θ, xs, prior_shocks)

    mean_std_expected_prices = generate_mean_std_pricing_decisions(df, policyF,
                                                                   lambdas0, min_periods)

    return np.linalg.norm(mean_std_expected_prices.values
                          - mean_std_observed_prices.values)


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
              prior_shocks: np.array, df: pd.DataFrame, min_periods: int = 3, w=None) -> float:
    """
    Computes the gmm error of the different between the observed moments and
    the moments predicted by the model + θ

    Moments: average (over firms) standard deviation for each time period

    xs: characteristics of firms
    mean_std_observed_prices: mean (over firms) of standard deviation per t
    w: weighting matrix for GMM objective
    """
    lambdas0 = from_theta_to_lambda_for_all_firms(θ, xs, prior_shocks)

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
    #TODO: simplify this. Pass to values,  no need for np.newaxis
    g = (1 / t) * (mean_std_expected_prices - mean_std_observed_prices)[:, np.newaxis]

    return (g.T @ w @ g)[0, 0]
