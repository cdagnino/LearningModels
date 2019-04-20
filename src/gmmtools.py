import numpy as np
from numba import njit
import pandas as pd
import src
from scipy import optimize
from .from_parameters_to_lambdas import logit, force_sum_to_1, reparam_lambdas, fun, jac


def gen_prior_shocks(nfirms, σerror=0.005):
    return np.random.normal(loc=0., scale=σerror, size=nfirms)


@njit()
def logistic(x):
    return 1/(1+np.e**(-x))

#σerror=0.005. np.random.normal(0, σerror)


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


#Add pass through H here
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

    def fun_(x):
        return fun(x, src.betas_transition, Eβ, H)

    #Numerical procedure to get lambda vector from H, Eβ
    sol = optimize.root(fun_, logit(starting_values), jac=jac_)

    lambdas_sol = force_sum_to_1(reparam_lambdas(sol.x))

    return lambdas_sol


def from_theta_to_lambda_for_all_firms(θ, xs, prior_shocks):
    nfirms = len(xs)
    lambdas0 = np.empty((nfirms, 3))
    for firm_i in range(nfirms):
        lambdas0[firm_i, :] = src.from_theta_to_lambda0(xs[firm_i], θ,
                                                       prior_shocks[firm_i])

    return lambdas0


# TODO: speed up this function. Can't jit it because policyF is a scipy LinearNDInterpolation f
def generate_pricing_decisions(policyF, lambda0: np.ndarray, demand_obs: np.ndarray) -> np.ndarray:
    """
    Generates a vector of pricing for one firm based on the policy function
    (could be vectorized later!)
    """
    current_lambdas = lambda0
    level_price_decisions = np.empty_like(demand_obs)
    for t, log_dmd in enumerate(demand_obs):
        level_price = policyF(current_lambdas[:2])  # Check: Is this correctly defined with the first two elements?
        level_price_decisions[t] = level_price

        # lambda updates: log_dmd: Yes, level_price: Yes
        new_lambdas = src.update_lambdas(log_dmd, src.dmd_transition_fs, current_lambdas,
                                         action=level_price, old_state=1.2)

        current_lambdas = new_lambdas

    return level_price_decisions


def generate_mean_std_pricing_decisions(df, policyF, lambdas_at_0, min_periods=3):
    """
    Lambdas0: starting priors for each of the N firms
    """
    pricing_decision_dfs = []
    for i, firm in enumerate(df.firm.unique()):
        prices = src.generate_pricing_decisions(policyF, lambdas_at_0[i],
                                                df[df.firm == firm].log_dmd.values)
        pricing_decision_dfs.append(pd.DataFrame({'level_prices': prices,
                                                  'firm': np.repeat(firm, len(prices))
                                                  }))

    pricing_decision_df = pd.concat(pricing_decision_dfs, axis=0)

    std_dev_df = (pricing_decision_df.groupby('firm').level_prices.rolling(window=4,
                                                                           min=min_periods)
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


def gmm_error(θ: np.array, policyF: object, xs: np.array, mean_std_observed_prices: pd.Series,
              prior_shocks: np.array, df: pd.DataFrame, min_periods: int = 3, w=None) -> float:
    """
    Computes the gmm error of the different between the observed moments and
    the moments predicted by the model + θ

    Moments: average (over firms) standard deviation for each time period

    x: characteristics of firms
    mean_std_observed_prices: mean (over firms) of standard deviation per t
    """
    lambdas0 = from_theta_to_lambda_for_all_firms(θ, xs, prior_shocks)
    mean_std_expected_prices = generate_mean_std_pricing_decisions(df, policyF,
                                                                   lambdas0, min_periods)

    #Eliminate nulls and get intersection of observed and expected moments
    mean_std_observed_prices = mean_std_observed_prices[pd.notnull(mean_std_observed_prices)]
    mean_std_expected_prices = mean_std_expected_prices[pd.notnull(mean_std_expected_prices)]
    index_inters = np.intersect1d(mean_std_observed_prices.index,
                                  mean_std_expected_prices.index)

    mean_std_observed_prices = mean_std_observed_prices.loc[index_inters]
    mean_std_expected_prices = mean_std_expected_prices.loc[index_inters]

    try:
        assert len(mean_std_observed_prices) == len(mean_std_expected_prices)
    except AssertionError as e:
        e.args += (len(mean_std_observed_prices), len(mean_std_expected_prices))
        raise
    t = len(mean_std_expected_prices)
    if w is None:
        w = np.identity(t)
    g = (1 / t) * (mean_std_expected_prices - mean_std_observed_prices)[:, np.newaxis]
    return (g.T @ w @ g)[0, 0]
