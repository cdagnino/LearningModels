import numpy as np
from src import const


#TODO: should be imported from aguirregabiria_simple.py
def period_profit(p: np.ndarray, lambdas: np.ndarray, betas_transition=const.betas_transition):
    """
    Correct expected period return profit. See ReadMe for derivation
    """
    constant_part = (p-const.c) * np.e ** const.α * np.e ** ((const.σ_ɛ ** 2) / 2)
    summation = np.dot(np.e**(betas_transition*np.log(p[:, np.newaxis])), lambdas)

    return constant_part*summation


def test_period_profit():
    p = np.array([1.4, 1.2])
    lambdas = np.array([0.5, 0.4, 0.1])

    beta_p_part = np.array([[np.e ** (-3. * 0.33647224), np.e ** (-2.5 * 0.33647224), np.e ** (-2 * 0.33647224)],
                            [np.e ** (-3. * 0.18232156), np.e ** (-2.5 * 0.18232156), np.e ** (-2 * 0.18232156)]])
    summation_part = np.array([0.36443148 * lambdas[0] + 0.43120115 * lambdas[1] + 0.51020408 * lambdas[2],
                               0.5787037 * lambdas[0] + 0.63393814 * lambdas[1] + 0.69444444 * lambdas[2]])

    expected = (p - const.c) * np.e ** const.α * np.e ** ((const.σ_ɛ ** 2) / 2) * summation_part

    computed = period_profit(p, lambdas)

    assert np.allclose(expected, computed, rtol=0.05)