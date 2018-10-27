import src
import numpy as np
import pandas as pd
from numba import njit


def generate_dmd_shocks(n: int, t: int, dmd_σϵ=src.const.σ_ɛ)\
                       -> np.ndarray:
    return np.random.normal(loc=0, scale=dmd_σϵ, size=(n, t))


def simulate_one_firm(valueF, policyF, dmd_shocks, prior_shock,
                      x, θ, true_beta=src.betas_transition[2]):
    """
    :param valueF: interpolated value function
    :param policyF: interpolated policy function
    :param maxt: maximum number of time periods
    :param dmd_shocks: fixed demand shocks (needed for GMM convergence)
    :param prior_shock: shock that determines relationship between theta and lambda
    :param x: characteristics of firm
    :param θ: structural parameters
    :param true_beta:
    :param dmd_σϵ:
    :return:
    """

    maxt = len(dmd_shocks)
    current_lambdas = src.from_theta_to_lambda0(x, θ, prior_shock)

    d = dict()
    d['level_prices'] = []
    d['log_dmd'] = []
    d['valueF'] = []
    d['lambda1'] = []
    d['lambda2'] = []
    d['lambda3'] = []
    d['t'] = []

    for t in range(maxt):
        d['t'].append(t)
        d['lambda1'].append(current_lambdas[0])
        d['lambda2'].append(current_lambdas[1])
        d['lambda3'].append(current_lambdas[2])
        d['valueF'].append(valueF(current_lambdas[:2])[0])

        # 0. Choose optimal price (last action of t-1)
        level_price = policyF(current_lambdas[:2])  # Check: Is this correctly defined with the first two elements?
        d['level_prices'].append(level_price[0])

        # 1. Demand happens
        log_dmd = src.draw_true_log_dmd(level_price, true_beta, dmd_shocks[t])
        d['log_dmd'].append(log_dmd[0])

        # 2. lambda updates: log_dmd: Yes, level_price: Yes
        new_lambdas = src.update_lambdas(log_dmd, src.dmd_transition_fs, current_lambdas,
                                         action=level_price, old_state=1.2)

        current_lambdas = new_lambdas

    return pd.DataFrame(d)



def simulate_all_firms(nfirms, valueF, policyF, xs, θ, dmd_shocks, prior_shocks, **kwargs):
    """
    Simulates data for all firms

    :param nfirms:
    :param valueF:
    :param policyF:
    :param xs:
    :param θ: true parameters that generate lambda0 and rest of data
    :param dmd_shocks:
    :param prior_shocks:
    :param kwargs:
    :return:
    """
    dfs = []
    for firm_i in range(nfirms):
        df = src.simulate_one_firm(valueF, policyF,
                                   dmd_shocks=dmd_shocks[firm_i, :],
                                   prior_shock=prior_shocks[firm_i],
                                   x=xs[firm_i], θ=θ, **kwargs)
        df['firm'] = firm_i
        dfs.append(df)

    return pd.concat(dfs, axis=0)