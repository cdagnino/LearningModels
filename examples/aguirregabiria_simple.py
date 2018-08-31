

import src
import src.constants as const
import numpy as np
import time
from typing import Callable
start = time.time()


length_of_price_grid = 10
min_price, max_price = 0.5, 1.5
n_of_lambdas_per_dim = 3
max_iters = 5
error_tol = 1e-5


def myopic_price(lambdas: np.ndarray, betas_transition=const.betas_transition):
    """
    Given a lambda point, spits out optimal myopic price
    """
    #Elasticity implied by lambdas
    elasticity = np.dot(lambdas, betas_transition) #-2.2
    assert elasticity < -1.0
    return const.c / (1 + (1/elasticity))


#TODO: get true expected value
def period_profit(p, lambdas, betas_transition=const.betas_transition):
    """
    Not the right expected profit (expected value doesn't make epsilon go away)
    but it should be close enough
    """
    E_β = src.exp_b_from_lambdas(lambdas, betas_transition)
    logq = const.α + E_β*np.log(p)
    return (p-const.c)*np.e**logq


def v0(lambdas_except_last: np.ndarray) -> Callable:
    """

    :param lambdas_except_last: D-1, then augmented
    :return:
    """
    full_lambdas = np.array(list(lambdas_except_last) + [1 - lambdas_except_last.sum()])
    optimal_price = myopic_price(full_lambdas)
    return period_profit(optimal_price, full_lambdas)


price_grid = np.linspace(min_price, max_price, num=length_of_price_grid)
simplex3d = src.generate_simplex_3dims(n_per_dim=n_of_lambdas_per_dim)


if __name__ == "__main__":
    v, policy, error = src.compute_fixed_point(src.bellman_operator, v0, error_tol=error_tol,
                                               max_iter=max_iters, verbose=1,
                                               skip=1, eval_grid=simplex3d[:, 0:2],
                                               price_grid=price_grid,
                                               lambda_simplex=simplex3d,
                                               period_return_f=period_profit)

    print("Error : ", error)


if __name__ == "__main2__":
    Tw, policy_t = src.bellman_operator(v0, price_grid, lambda_simplex=simplex3d)
    print(Tw)
    print("========")
    print(policy_t)

    print("Done in ", (time.time() - start)/60,
          " minutes. Lambda simplex had {0} dimensions".format(len(simplex3d)))

    print("========")
    Tw, policy_t = src.bellman_operator(Tw, price_grid, lambda_simplex=simplex3d)
    print(Tw)
    print("========")
    print(policy_t)

    print("Done in ", (time.time() - start)/60,
          " minutes. Lambda simplex had {0} dimensions".format(len(simplex3d)))
