import numpy as np
from numba import njit


def gen_prior_shocks(nfirms, σerror=0.005):
    return np.random.normal(loc=0., scale=σerror, size=nfirms)


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


@njit()
def from_theta_to_lambda0(x, θ, prior_shock):
    """
    Generates a lambda0 vector from the theta vector and x
    θ = [θ10, θ11, θ20, θ21]
    x : characteristics of firms
    prior_shock: puts randomness in the relationship between theta and lambda
    """
    lambda1 = logistic(θ[0] + θ[1]*x + prior_shock)
    maxlambda2_value = 1 - lambda1
    #np.clip ---> nb_clip
    lambda2 = nb_clip(logistic(θ[2] + θ[3]*x + prior_shock),
                      0, maxlambda2_value)
    lambda3 = logistic(1 - lambda1 - lambda2)
    return np.array([lambda1, lambda2, lambda3])