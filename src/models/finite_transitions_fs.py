"""
See Aguirregabiria & Jeon (2018), around equation (2)

$V_{b_t}(I_t)$ is the value of the firm at period $t$ given current information and beliefs.

The value of the firm is given by

$$V_{b_t}(I_t) = max_{a_t \in A} \{ \pi(a_t, x_t) + \beta
                 \int V_{b_{t+1}}(x_{t+1}, I_t) b_t(x_{t+1}| a_t, I_t )\; d x_{t+1}\}  $$

Probably better notation would be to write $V_{b_{t+1}}(I_{t+1}(x_{t+1}, I_t))$

the firm has a prior belief $b_0(x_1 | a_0, x_0)$ that is exogenous.
This prior is a mixuter over a collection of L transition probabilities:

$$P = \{p_l (x_{t+1} | a_t, x_t)  \}_{l=1}^L$$

so that

$$b_0(x_1 | a_0, x_0) = \sum_{l=1}^L \lambda_l^{(0)} p_l (x_1| a_0, x_0)$$

The firm observes the new state $x_t$ and uses this information to update its beliefs by using Bayes rule. The Bayesian updating is given by

$$\lambda_l^{(t)} = \frac{ p_l (x_t| a_{t-1}, x_{t-1}) \lambda_l^{(t-1)} }{      \sum_{l'=1}^L p_{l'} (x_t| a_{t-1}, x_{t-1}) \lambda_{l'}^{(t-1)}} $$

In words, $p_l (x_t| a_{t-1}, x_{t-1})$ is the probability that the $l$
transition probability gave to $x_t$ actually happening. If the probability of $x_t$ (the state that actually occured) is high under $l$, then that $l$ transition probability will get a higher weight in the beliefs of next period.
"""


import src.constants as const
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
from typing import Callable
from scipy.interpolate import LinearNDInterpolator


#TODO: vectorize over action (price). Hadamard + dot. Check black notebook
def belief(new_state, transition_fs, lambda_weights, action: float, old_state) -> float:
    """
    state: point in state space
    transition_fs: list of transition probabilities
    """
    return np.dot(transition_fs(new_state, action, old_state), lambda_weights)


def update_lambdas(new_state: float, transition_fs: Callable, old_lambdas: np.ndarray,
                   action: float, old_state) -> np.ndarray:
    """
    Update the beliefs for new lambdas given a new state.
    Transition_fs are fixed and exogenous. k by 1

    Output:
    """
    denominator = (old_lambdas * transition_fs(new_state, action, old_state)).sum()
    assert isinstance(denominator, float)
    return transition_fs(new_state, action, old_state)*old_lambdas / denominator


def dmd_transition_fs(new_state, action: float, old_state) -> np.ndarray:
    """
    Returns the probability of observing a given log demand
    Action is the price
    """
    return np.array([norm.pdf(new_state, loc=(const.α + beta*np.log(action)),
                              scale=const.σ_ɛ) for beta in const.betas_transition])


def exp_b_from_lambdas(lambdas, betas_transition=const.betas_transition):
    """
    Get E[β] from the lambdas
    """
    return np.dot(lambdas, betas_transition)


def eOfV(wGuess, p_array, lambdas: np.ndarray) -> np.ndarray:
    """
    Integrates wGuess * belief_f for each value of p. Integration over demand

    Sum of points on demand and weights, multiplied by V and the belief function
    """
    #TODO: vectorize over p
    integrated_values = np.empty(p_array.shape[0])
    for i in range(len(integrated_values)):
        def new_lambdas(new_dmd):
            return update_lambdas(new_dmd, transition_fs=dmd_transition_fs,
                                  old_lambdas=lambdas, action=p_array[i], old_state='No worries')

        def new_belief(new_dmd):
            return belief(new_dmd, transition_fs=dmd_transition_fs,
                          lambda_weights=new_lambdas(new_dmd),
                          action=p_array[i], old_state='No worries')

        #wGuess takes all lambdas except last
        integrand = lambda new_dmd: wGuess(new_lambdas(new_dmd)[:-1]) * new_belief(new_dmd)

        #TODO: check if these limits are for D, not for logD!!
        integrated_values[i], error = integrate.quad(integrand, 0.001, 10)
        if i % 30 == 0:
            print("error integración: ", error)

    return integrated_values


def interpolate_wguess(simplex_points,
                       value_function_points: np.ndarray) -> Callable:
    """
    :param simplex_points: defined on D dims. Interpolation happens on D-1 dims
    :param value_function_points:
    :return:
    """
    dims = simplex_points.shape[1]
    return LinearNDInterpolator(simplex_points[:, 0:(dims - 1)], value_function_points)


def bellman_operator(wGuess, price_grid, lambda_simplex, period_return_f: Callable):
    """
    The approximate Bellman operator, which computes and returns the
    updated value function Tw on the grid points.

    :param wGuess: Matrix on lambdas or function on lambdas
    :param price_grid:
    :param lambda_simplex:
    :param period_return_f: Period return function. E.g., current period profit
    :return: interpolated_tw, policy
    """

    # policy = np.empty_like(wGuess)
    # Tw = np.empty_like(wGuess)
    policy = np.empty(lambda_simplex.shape[0])
    Tw = np.empty(lambda_simplex.shape[0])

    # 1. go over grid of state space
    # 2. Write objective (present return + delta*eOfV)
    # 3. Find optimal p on that objective
    # 4. write optimal p and value function on that point in the grid
    for iII, (λ1, λ2, λ3) in enumerate(lambda_simplex):
        print("doing {0} of {1}".format(iII, len(lambda_simplex)))
        lambda_weights = np.array([λ1, λ2, λ3])

        R_ : np.ndarray = period_return_f(price_grid, lambdas=lambda_weights)
        eOfV_p : np.ndarray = eOfV(wGuess, price_grid, lambdas=lambda_weights)
        assert isinstance(R_, np.ndarray)
        assert isinstance(eOfV_p, np.ndarray)
        objective_vals = R_ + const.δ * eOfV_p
        p_argmax = np.argmax(objective_vals)
        pStar = price_grid[p_argmax]
        policy[iII] = pStar
        Tw[iII] = objective_vals[p_argmax]

    interpolated_tw = interpolate_wguess(lambda_simplex, Tw)
    return interpolated_tw, policy







