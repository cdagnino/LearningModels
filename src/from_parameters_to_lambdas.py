import src
import numpy as np
from scipy.stats import entropy
from scipy.special import expit
from numba import njit


def my_entropy(p):
    return entropy(p)


@njit()
def force_sum_to_1(orig_lambdas):
    """
    Forces lambdas to sum to 1
    (although last element might be negative)
    """
    sum_lambdas = orig_lambdas.sum()
    if sum_lambdas > 1.:
        orig_lambdas /= sum_lambdas
        # TODO: think if this is what I want: might make third lambda 0 too much
        return np.concatenate((orig_lambdas, np.array([0.])))
    else:
        return np.concatenate((orig_lambdas, 1 - np.array([sum_lambdas])))


def logit(p):
    return np.log(p / (1 - p))


def reparam_lambdas(x):
    """ inverse logit. Forces the lambdas to be within 0 and 1"""
    return expit(x)


#@njit()
def h_and_exp_betas_eqns(orig_lambdas, βs, Eβ, H, w=np.array([[1., 0.], [0., 1./4.]])):
    """
    orig_lambdas: original lambda tries (not summing to zero, not within [0, 1])
    Eβ, H: the objectives
    βs: fixed constant of the model
    """
    lambdas = force_sum_to_1(src.reparam_lambdas(orig_lambdas))
    g = np.array([entropy(lambdas) - H, np.dot(βs, lambdas) - Eβ])
    return g.T @ w @ g


#Not relevant anymore (minimize is using a derivative free method)
def jac(x, βs):
    """
    Jacobian for reparametrization of lambdas.
    Code for only three lambdas
    """
    # Derivatives wrt to H
    block = np.log((1 - np.e ** (x[0] + x[1])) / (np.e ** (x[0]) + np.e ** (x[1]) + np.e ** (x[0] + x[1]) + 1))
    num0 = (-np.log(np.e ** x[0] / (np.e ** x[0] + 1)) + block) * np.e ** x[0]
    den0 = np.e ** (2 * x[0]) + 2 * np.e ** (x[0]) + 1
    num1 = (-np.log(np.e ** x[1] / (np.e ** x[1] + 1)) + block) * np.e ** x[1]
    den1 = np.e ** (2 * x[1]) + 2 * np.e ** (x[1]) + 1

    dh_dx = np.array([num0 / den0, num1 / den1])

    # Derivatives wrt E[B]
    deb_0 = ((βs[0] - βs[2]) * np.e ** (-x[0])) / (1 + np.e ** (-x[0])) ** 2
    deb_1 = ((βs[1] - βs[2]) * np.e ** (-x[1])) / (1 + np.e ** (-x[1])) ** 2
    deb_dx = np.array([deb_0, deb_1])

    return np.array([dh_dx, deb_dx])


def relative_error(true, solution):
    """
    Average relative error to discriminate solutions
    """
    return np.mean(2 * np.abs((true - solution) / (true + solution)))