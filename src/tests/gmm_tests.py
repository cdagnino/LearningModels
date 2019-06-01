import pandas as pd
import src
import numpy as np
from numba import njit

np.random.seed(28281111)


# FIRST: assume all firms have the same size
# THEN: generalize if needed


def inner_loop_with_pandas(simul_repetitions, df, γ, taste_shocks, b0):
    """
    Return an len(df) by M matrix of betas_inertia values
    """

    betas_inertia_by_m = np.empty((len(df), simul_repetitions))
    for m in range(simul_repetitions):
        for i_firm, firm in enumerate(df.firm.unique()):
            mask: pd.Series = (df.firm == firm)
            t = mask.sum()
            df.loc[mask, "betas_inertia"] = src.generate_betas_inertia_Ξ(γ, taste_shocks,
                                                                         b0, t, i_firm)
        betas_inertia_by_m[:, m] = df.betas_inertia.values

    return betas_inertia_by_m


@njit()
def inner_loop_with_numba(simul_repetitions, firm_length, n_firms, γ, taste_shocks, b0):
    betas_inertia_by_m = np.empty((firm_length * n_firms, simul_repetitions))
    for m in range(simul_repetitions):
        for i_firm in range(n_firms):
            betas_inertia_by_m[firm_length * i_firm:firm_length * (i_firm + 1), m] = src.generate_betas_inertia_Ξ(γ,
                                                                                                                  taste_shocks,
                                                                                                                  b0,
                                                                                                                  firm_length,
                                                                                                                  i_firm)
    return betas_inertia_by_m


@njit()
def inner_loop_with_numba_unbalanced(simul_repetitions, firm_lengths: np.array, n_firms: int, len_df: int, γ,
                                     taste_shocks, b0):
    """
    firm_lengths is an array that marks the location where each firm ends, prepended by a 0
    """
    betas_inertia_by_m = np.empty((len_df, simul_repetitions))
    for m in range(simul_repetitions):
        for i_firm in range(n_firms):
            betas_inertia_by_m[firm_lengths[i_firm]:firm_lengths[i_firm + 1], m] = \
                (src.generate_betas_inertia_Ξ(γ, taste_shocks, b0,
                                              firm_lengths[i_firm + 1] - firm_lengths[i_firm],
                                              i_firm))
    return betas_inertia_by_m


def test_inner_simulation_loop():
    """
    :return:
    """

    # Setup

    df = pd.DataFrame({'firm': [1, 1, 1, 1,
                                2, 2, 2, 2],
                       'log_dmd': [12, 11, 13, 13,
                                   21, 20, 20, 20]})

    simul_repetitions = 10
    γ = 0.7

    n_firms = df.firm.nunique()
    max_t_periods_in_data = df.groupby('firm').log_dmd.count().max()
    firm_length = max_t_periods_in_data
    # prior_shocks = src.gen_prior_shocks(n_firms, σerror=0)
    taste_shocks = np.random.normal(loc=0, scale=1, size=(max_t_periods_in_data, n_firms))
    b0 = np.random.normal(loc=0, scale=1, size=n_firms)

    with_pandas = inner_loop_with_pandas(simul_repetitions, df, γ, taste_shocks, b0)
    with_numba = inner_loop_with_numba(simul_repetitions, firm_length, n_firms, γ, taste_shocks, b0)

    # return with_pandas, with_numba
    assert (with_numba == with_pandas).all()


def test_inner_simulation_loop_unbalanced():
    """
    :return:
    """

    # Setup: unbalanced df
    df = pd.DataFrame({'firm': [1, 1, 1, 1, 1,
                                2, 2, 2, 2],
                       'log_dmd': [12, 11, 13, 13, 14,
                                   21, 20, 20, 20]})

    simul_repetitions = 10
    γ = 0.7

    n_firms = df.firm.nunique()
    len_df = len(df)
    max_t_periods_in_data = df.groupby('firm').log_dmd.count().max()
    firm_length = max_t_periods_in_data
    # prior_shocks = src.gen_prior_shocks(n_firms, σerror=0)
    taste_shocks = np.random.normal(loc=0, scale=1, size=(max_t_periods_in_data, n_firms))
    b0 = np.random.normal(loc=0, scale=1, size=n_firms)

    with_pandas = inner_loop_with_pandas(simul_repetitions, df, γ, taste_shocks, b0)

    all_firms = df.firm.unique()
    firm_lengths = np.empty(len(all_firms) + 1, dtype=int)
    firm_lengths[0] = 0
    for i, firm_i in enumerate(all_firms):
        firm_lengths[i + 1] = df[df.firm == firm_i].index[-1] + 1
    with_numba = inner_loop_with_numba_unbalanced(simul_repetitions, firm_lengths, n_firms, len_df, γ, taste_shocks, b0)

    assert (with_numba == with_pandas).all()