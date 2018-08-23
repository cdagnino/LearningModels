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


import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
from typing import Callable
from scipy.interpolate import LinearNDInterpolator


##########
#
# Constants
#
##########

betas_transition = [-3.0, -2.5, -2.0]
σ_ɛ = 0.5
α = 1.0
c = 0.5
δ = 0.9
n_per_dim = 20
k = 3 #Number of dimensions of lambdas vector


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
    Give the probability of observation a given log demand
    Action is the price
    """
    return np.array([norm.pdf(new_state, loc=(α + beta*np.log(action)),
                              scale=σ_ɛ)  for beta in betas_transition  ])


def test_update_lambdas():
    new_lambdas = update_lambdas(2.1, dmd_transition_fs, np.array([0., 0., 1.]),
                                 action=0.8, old_state="meh")
    assert np.isfinite(new_lambdas).all()

test_update_lambdas()

def test_belief():
    belief_at_x = belief(3.1, dmd_transition_fs,
                         [0., 0., 1.], 3.0, old_state='meh')
    assert isinstance(belief_at_x, float)
    assert np.isfinite(belief_at_x)

test_belief()


def exp_b_from_lambdas(lambdas, betas_transition=betas_transition):
    """
    Get E[β] from the lambdas
    """
    return np.dot(lambdas, betas_transition)


def myopic_price(lambdas: np.ndarray, betas_transition=betas_transition):
    """
    Given a lambda point, spits out optimal myopic price
    """
    #Elasticity implied by lambdas
    elasticity = np.dot(lambdas, betas_transition) #-2.2
    try:
        assert elasticity < -1.0
    except:
        print("hola bu")
    return c / (1 + (1/elasticity))


#TODO: get true expected value
def period_profit(p, lambdas, betas_transition=betas_transition):
    """
    Not the right expected profit (expected value doesn't make epsilon go away)
    but it should be close enough
    """
    E_β = exp_b_from_lambdas(lambdas, betas_transition)
    logq = α + E_β*np.log(p)
    return (p-c)*np.e**(logq)


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


def bellman_operator(wGuess, price_grid, lambda_simplex):
    """
    The approximate Bellman operator, which computes and returns the
    updated value function Tw on the grid points.

    :param wGuess: Matrix on lambdas or function on lambdas
    :param price_grid:
    :param lambda_simplex:
    :return: interpolated_tw, policy
    """

    # policy = np.empty_like(wGuess)
    # Tw = np.empty_like(wGuess)
    policy = np.empty(simplex3d.shape[0])
    Tw = np.empty(simplex3d.shape[0])

    # 1. go over grid of state space
    # 2. Write objective (present return + delta*eOfV)
    # 3. Find optimal p on that objective
    # 4. write optimal p and value function on that point in the grid
    for iII, (λ1, λ2, λ3) in enumerate(lambda_simplex):
        print("doing {0} of {1}".format(iII, len(lambda_simplex)))
        lambda_weights = np.array([λ1, λ2, λ3])

        R_ : np.ndarray = period_profit(price_grid, lambdas=lambda_weights)
        eOfV_p : np.ndarray = eOfV(wGuess, price_grid, lambdas=lambda_weights)
        assert isinstance(R_, np.ndarray)
        assert isinstance(eOfV_p, np.ndarray)
        objective_vals = R_ + δ * eOfV_p
        p_argmax = np.argmax(objective_vals)
        pStar = price_grid[p_argmax]
        policy[iII] = pStar
        Tw[iII] = objective_vals[p_argmax]

    interpolated_tw = interpolate_wguess(lambda_simplex, Tw)
    return interpolated_tw, policy


############
#
# Short script
#
##########

import time
start = time.time()


# TODO send to utils
def generate_simplex_3dims(n_per_dim=20):
    xlist = np.linspace(0.0, 1.0, n_per_dim)
    ylist = np.linspace(0.0, 1.0, n_per_dim)
    zlist = np.linspace(0.0, 1.0, n_per_dim)
    return np.array([[x, y, z] for x in xlist for y in ylist for z in zlist
                     if np.allclose(x+y+z, 1.0)])


def v0(lambdas_except_last: np.ndarray) -> Callable:
    """

    :param lambdas_except_last: D-1, then augmented
    :return:
    """
    full_lambdas = np.array(list(lambdas_except_last) + [1 - lambdas_except_last.sum()])
    optimal_price = myopic_price(full_lambdas)
    return period_profit(optimal_price, full_lambdas)


price_grid = np.linspace(0.5, 1.5, num=10)
simplex3d = generate_simplex_3dims(n_per_dim=3)




def compute_fixed_point(T, v, error_tol=1e-5, max_iter=50, verbose=1,
                        skip=10, eval_grid=None, *args,
                        **kwargs):
    """
    Computes and returns :math:`T^k v`, an approximate fixed point.

    Here T is an operator, v is an initial condition and k is the number
    of iterates. Provided that T is a contraction mapping or similar,
    :math:`T^k v` will be an approximation to the fixed point.

    Parameters
    ----------
    T : callable
        A callable object (e.g., function) that acts on v.
        The bellman operator
    v : object
        An object such that T(v) is defined. Initial guess.
    error_tol : scalar(float), optional(default=1e-5)
        Error tolerance
    max_iter : scalar(int), optional(default=50)
        Maximum number of iterations
    verbose : bool, optional(default=True)
        If True then print current error at each iterate.
    args, kwargs :
        Other arguments and keyword arguments that are passed directly
        to  the function T each time it is called

    Returns
    -------
    v : object (usually array)
        The approximate fixed point
    policy : object (usually array)
        the policy at the approximate fixed point

    """
    iterate = 0
    error = error_tol + 1
    while iterate < max_iter and error > error_tol:
        start_iter = time.time()
        new_v, policy = T(v, *args, **kwargs)
        iterate += 1
        try:
            error = np.max(np.abs(new_v - v))
        except TypeError:
            #Calculate error for functions
            n_eval_grid = len(eval_grid)
            new_v_evals = np.empty(n_eval_grid)
            v_evals = np.empty(n_eval_grid)
            for i in range(n_eval_grid):
                new_v_evals = new_v(eval_grid[i])
                v_evals = v(eval_grid[i])
            error = np.max(np.abs(new_v_evals - v_evals))
        if verbose and iterate % skip == 0:
            time_taken = (time.time() - start_iter) / 60.
            print("Computed iterate %d with error %f in %f minutes" % (iterate, error, time_taken))
        try:
            v[:] = new_v
        except TypeError:
            v = new_v
    return v, policy, error


if __name__ == "__main__":
    v, policy, error = compute_fixed_point(bellman_operator, v0, error_tol=1e-5, max_iter=5, verbose=1,
                        skip=1, eval_grid=simplex3d[:, 0:2], price_grid=price_grid,
                        lambda_simplex=simplex3d)

    print("Error : ", error)


if __name__ == "__main2__":
    Tw, policy_t = bellman_operator(v0, price_grid, lambda_simplex=simplex3d)
    print(Tw)
    print("========")
    print(policy_t)

    print("Done in ", (time.time() - start)/60,
          " minutes. Lambda simplex had {0} dimensions".format(len(simplex3d)))

    print("========")
    Tw, policy_t = bellman_operator(Tw, price_grid, lambda_simplex=simplex3d)
    print(Tw)
    print("========")
    print(policy_t)

    print("Done in ", (time.time() - start)/60,
          " minutes. Lambda simplex had {0} dimensions".format(len(simplex3d)))

