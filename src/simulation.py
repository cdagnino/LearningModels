import src
import numpy as np
import pandas as pd


def simulate_one_firm(valueF, policyF, maxt, lambda0=np.array([0.4, 0.4, 0.2]),
            true_beta=src.betas_transition[2],
            dmd_σϵ=src.const.σ_ɛ):
    """
    Simulates the action of one firm when facing a random demand

    :param valueF: interpolated value function
    :param policyF: interpolated policy function
    :param maxt: maximum number of time periods
    :param lambda0: starting prior
    :param true_beta: true elasticity of demand
    :param dmd_σϵ: standard deviation of demand noise
    :return: pd.Dataframe with level prices, log_dmd, value function and the lambdas
    """
    current_lambdas = lambda0
    d = {}
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
        log_dmd = src.draw_true_log_dmd(level_price, true_beta, dmd_σϵ)
        d['log_dmd'].append(log_dmd[0])

        # 2. lambda updates: log_dmd: Yes, level_price: Yes
        new_lambdas = src.update_lambdas(log_dmd, src.dmd_transition_fs, current_lambdas,
                                         action=level_price, old_state=1.2)

        current_lambdas = new_lambdas

    return pd.DataFrame(d)


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

