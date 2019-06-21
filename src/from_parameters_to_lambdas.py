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
        return np.concatenate((orig_lambdas, np.array([sum_lambdas])))


def logit(p):
    return np.log(p / (1 - p))


def reparam_lambdas(x):
    """use softmax"""
    #return np.exp(x) / np.sum(np.exp(x))
    #Numerically more stable version
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


#@njit()
def h_and_exp_betas_eqns(orig_lambdas, βs, Eβ, H, w=np.array([[1., 0.], [0., 1./4.]])):
    """
    orig_lambdas: original lambda tries (not summing to zero, not within [0, 1])
    Eβ, H: the objectives
    βs: fixed constant of the model
    """
    lambdas = src.reparam_lambdas(orig_lambdas)
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


def input_heb_to_lambda_con_norm(input_point, lambda_values, h_candidates, eb_candidates,
                                 h_n_digits_precision, eb_n_digits_precision,
                                 h_dict, e_dict, interpolate=True):
    """
    Receives an input_point in H and E[B] space (plus precalculated lambdas)
    and spits the corresponding lambda0 solution

    :param input_point:
    :param lambda_values:
    :param h_candidates:
    :param eb_candidates:
    :param h_n_digits_precision:
    :param eb_n_digits_precision:
    :param h_dict:
    :param e_dict:
    :param interpolate: whether to interpolate or just use closest value
    :return:
    """
    H, eb = input_point[0], input_point[1]
    # Find row, col
    row = h_dict[np.round(H, h_n_digits_precision - 1)]
    col = e_dict[np.round(eb, eb_n_digits_precision - 1)]

    if interpolate:
        # Distances to row-1, row, row+1 and col-1, col, col+1
        dist_row = np.array([np.abs(H - h_candidates[row - 1]), np.abs(H - h_candidates[row]),
                             np.abs(H - h_candidates[row + 1])])
        if np.argmax(dist_row) == 2:
            relevant_rows = [row - 1, row]
        else: # np.argmax(dist_row) == 0:
            relevant_rows = [row, row + 1]

        dist_col = np.array([np.abs(eb - eb_candidates[col - 1]), np.abs(eb - eb_candidates[col]),
                             np.abs(eb - eb_candidates[col + 1])])
        if np.argmax(dist_col) == 2:
            relevant_cols = [col - 1, col]
        else: #np.argmax(dist_col) == 0:
            relevant_cols = [col, col + 1]

        pointA, pointB = [relevant_rows[0], relevant_cols[0]], [relevant_rows[0], relevant_cols[1]]
        pointC, pointD = [relevant_rows[1], relevant_cols[0]], [relevant_rows[1], relevant_cols[1]]
        points = [pointA, pointB, pointC, pointD]

        pointA_value = lambda_values[relevant_rows[0], relevant_cols[0]]
        pointB_value = lambda_values[relevant_rows[0], relevant_cols[1]]
        pointC_value = lambda_values[relevant_rows[1], relevant_cols[0]]
        pointD_value = lambda_values[relevant_rows[1], relevant_cols[1]]
        values = np.array([pointA_value, pointB_value, pointC_value, pointD_value])
        distances = np.array([np.linalg.norm(input_point - point) for point in points])
        distances /= distances.sum()

        # Linear combination of A, B, C, D points
        return distances[np.newaxis, :] @ values
    else:
        return lambda_values[row, col]


@njit()
def numba_clip(x, a, b):
    if x < a:
        return a
    elif x > b:
        return b
    else:
        return x



@njit(['float64(float64, float64, float64, float64, float64[:])'])
def lambda_gen(h, eb, h2, eb2, lambdacoeffs):
    """
    Generates one lambda using the precalculated lambda coeffs
    """
    return numba_clip(lambdacoeffs[0]*h + lambdacoeffs[1]*eb +
                      lambdacoeffs[2]*h*eb + lambdacoeffs[3]*h2
                      + lambdacoeffs[4]*eb2 + lambdacoeffs[5]*(h2*eb2)
                      + lambdacoeffs[6]*(h2*eb), 0., 1.)


@njit(['float64[:](float64, float64, float64[:], float64[:], float64[:])'])
def all_lambdas_gen(h, eb, lcoeffs1, lcoeffs2, lcoeffs3):
    """
    Takes h, eb values and uses lambda_gen to return lambda1, lambda2, lambda3
    """
    h2 = h**2
    eb2 = eb**2
    return np.array([lambda_gen(h, eb, h2, eb2, lcoeffs1),
                    lambda_gen(h, eb, h2, eb2, lcoeffs2),
                    lambda_gen(h, eb, h2, eb2, lcoeffs3)])
