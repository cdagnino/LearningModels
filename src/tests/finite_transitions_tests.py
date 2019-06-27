import numpy as np
import src
from src import const
from scipy.stats import norm
from scipy import integrate
from typing import Callable
from src.utils import myopic_price


def test_update_lambdas():
    new_lambdas = src.update_lambdas(2.1, src.dmd_transition_fs, np.array([0., 0., 1.]),
                                     action=0.8, old_state=2.5)
    assert np.isfinite(new_lambdas).all()


def test_belief():
    belief_at_x = src.belief(3.1, src.dmd_transition_fs,
                             np.array([0., 0., 1.]), 3.0, old_state=2.5)
    assert isinstance(belief_at_x, float)
    assert np.isfinite(belief_at_x)

    belief_at_x = src.belief(2.1, src.dmd_transition_fs,
                             np.array([0.3, 0.4, 0.3]), 2.8, old_state=2.5)
    assert isinstance(belief_at_x, float)
    assert np.isfinite(belief_at_x)


def test_jitted_normpdf():
    x = 0.3
    loc, scale = 8.1, 1.5
    jit_result = src.jittednormpdf(x, loc, scale)
    scipy_result = norm.pdf(x, loc=loc, scale=scale)
    assert np.allclose(jit_result, scipy_result)


def test_dmd_transition_fs():
    def old_dmd_transition_fs(new_state, action: float, old_state) -> np.ndarray:
        """
        Returns the probability of observing a given log demand
        Action is the price
        """
        return np.array([src.jittednormpdf(new_state, loc=const.α + beta * np.log(action), scale=const.σ_ɛ)
                         for beta in const.betas_transition])

    new_state = 1.3
    action, old_state = 0.9, 1-5
    old_values = old_dmd_transition_fs(new_state, action, old_state)
    new_values = src.dmd_transition_fs(new_state, action, old_state)
    assert np.allclose(old_values, new_values)


def test_eofV():
    """
    Tests the integration of V over the possible x' values
    Compares it to a simple trapezoidal rule
    """

    def period_profit(p: np.ndarray, lambdas: np.ndarray, betas_transition=const.betas_transition):
        """
        Correct expected period return profit. See ReadMe for derivation
        """
        constant_part = (p - const.c) * np.e ** const.α * np.e ** ((const.σ_ɛ ** 2) / 2)
        summation = np.dot(np.e ** (betas_transition * np.log(p[:, np.newaxis])), lambdas)

        return constant_part * summation

    def v0(lambdas_except_last: np.ndarray) -> Callable:
        """

        :param lambdas_except_last: D-1, then augmented
        :return:
        """
        full_lambdas = np.array(list(lambdas_except_last) + [1 - lambdas_except_last.sum()])
        optimal_price: float = myopic_price(full_lambdas)

        # Dirty trick because period_profit takes a vector price, not scalar
        prices = np.array([optimal_price, optimal_price + 0.01])
        return period_profit(prices, full_lambdas)[0]


    #Setup
    length_of_price_grid = 10
    min_price, max_price = 0.5, 1.5
    p_array = np.linspace(min_price, max_price, num=length_of_price_grid)
    lambda_point = np.array([0.55555556, 0.11111111, 0.33333333])

    # Expected on trapezoid rule
    expected_results = np.empty_like(p_array)
    for i, price in enumerate(p_array):
        def new_lambdas(new_dmd):
            return src.update_lambdas(new_dmd, transition_fs=src.dmd_transition_fs,
                                      old_lambdas=lambda_point, action=price,
                                      old_state=2.5)

        def new_belief(new_dmd):
            """
            Don't update lambdas! Use the ones from the current period
            """
            return src.belief(new_dmd, transition_fs=src.dmd_transition_fs,
                              lambda_weights=lambda_point,
                              action=price, old_state=2.5)

        def integrand(new_dmd):
            return v0(new_lambdas(new_dmd)[:-1]) * new_belief(new_dmd)

        yvals = []
        xvals = np.linspace(-6, 5, num=20)
        for x in xvals:
            yvals.append(integrand(x))
        expected_results[i] = integrate.trapz(yvals, xvals)


    #Expected gauss-hermite
    ghermite_result = src.eOfV(v0, p_array, lambda_point)

    assert np.allclose(ghermite_result, expected_results, rtol=0.05)

