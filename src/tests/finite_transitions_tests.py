import numpy as np
import src
from src import const
from scipy.stats import norm


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

