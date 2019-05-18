import pandas as pd
import src
import numpy as np
from numba import njit

def inner_loop_with_pandas(simul_repetitions, df, γ, taste_shocks, b0):
    for m in range(simul_repetitions):
        for i_firm, firm in enumerate(df.firm.unique()):
            mask: pd.Series = (df.firm == firm)
            t = mask.sum()
            df.loc[mask, "betas_inertia"] = src.generate_betas_inertia_Ξ(γ, taste_shocks,
                                                                         b0, t, i_firm)

    return df


def inner_loop_with_numba(simul_repetitions, df, γ, taste_shocks, b0):
    for m in range(simul_repetitions):
        for i_firm, firm in enumerate(df.firm.unique()):
            mask: pd.Series = (df.firm == firm)
            t = mask.sum()
            df.loc[mask, "betas_inertia"] = src.generate_betas_inertia_Ξ(γ, taste_shocks,
                                                                         b0, t, i_firm)
    return df




def test_inner_simulation_loop():
    """
    :return:
    """

    #Setup
    df = pd.DataFrame({'firm': [1, 1, 1, 1, 1,
                                     2, 2, 2, 2],
                            'log_dmd': [12, 11, 13, 13, 14,
                                        21, 20, 20, 20]})

    simul_repetitions = 10
    γ = 0.7

    n_firms = df.firm.nunique()
    max_t_periods_in_data = df.groupby('firm').log_dmd.count().max()

    #prior_shocks = src.gen_prior_shocks(n_firms, σerror=0)
    taste_shocks = np.random.normal(loc=0, scale=1, size=(max_t_periods_in_data, n_firms))
    b0 = np.random.normal(loc=0, scale=1, size=n_firms)

    with_pandas = inner_loop_with_pandas(simul_repetitions, df, γ, taste_shocks, b0)
    with_numba = inner_loop_with_numba(simul_repetitions, df, γ, taste_shocks, b0)

    assert (with_pandas == with_numba).all()