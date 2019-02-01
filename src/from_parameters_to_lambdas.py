import src
import numpy as np
from numba import njit


#@njit()
def my_entropy(p):
    eps = 10e-9
    return -np.sum(p+eps * np.log(p+eps))


def force_sum_to_1(x):
    """
    Forces lambdas to sum to 1
    (although last element might be negative)
    """
    return np.hstack([x, 1-x.sum()])


def logit(p):
    return np.log(p / (1 - p))


def reparam_lambdas(x):
    """ inverse logit. Forces the lambdas to be within 0 and 1"""
    return np.e**x / (1 + np.e**x)


def fun(x, βs, Eβ, H):
    """
    x: deep parameters
    Eβ, H: the objectives
    βs: fixed constant of the model
    """
    lambdas = force_sum_to_1(reparam_lambdas(x))
    return [my_entropy(lambdas) - H,
            np.dot(βs, lambdas) - Eβ]


#TODO: make this general, not limited to dim(x)=3
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