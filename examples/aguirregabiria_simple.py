

import src
import src.constants as const
import numpy as np
import time, datetime
from typing import Callable
import dill
from numba.decorators import njit
start = time.time()

benchmark_values = False

#New benchmark: #0.02, 0.019, 0.019
if benchmark_values:
    length_of_price_grid = 10
    min_price, max_price = 0.5, 1.5
    n_of_lambdas_per_dim = 3
    max_iters = 20
    error_tol = 1e-5
else: #Time per iteration: 0.5, 4, 3.6, 4.4
    length_of_price_grid = 30
    min_price, max_price = 0.5, 6
    n_of_lambdas_per_dim = 15
    max_iters = 60
    error_tol = 1e-5


def period_profit(p: np.ndarray, lambdas: np.ndarray, betas_transition=const.betas_transition):
    """
    Correct expected period return profit. See ReadMe for derivation

    p: level prices, NOT log
    """
    constant_part = (p-const.c) * np.e ** const.α * np.e ** ((const.σ_ɛ ** 2) / 2)
    summation = np.dot(np.e**(betas_transition*np.log(p[:, np.newaxis])), lambdas)

    return constant_part*summation


def myopic_price(lambdas: np.ndarray, betas_transition=const.betas_transition):
    """
    Given a lambda point, spits out optimal myopic price
    """
    #Elasticity implied by lambdas
    elasticity = np.dot(lambdas, betas_transition) #-2.2
    assert elasticity < -1.0
    return const.c / (1 + (1/elasticity))


def v0(lambdas_except_last: np.ndarray) -> Callable:
    """

    :param lambdas_except_last: D-1, then augmented
    :return:
    """
    full_lambdas = np.array(list(lambdas_except_last) + [1 - lambdas_except_last.sum()])
    optimal_price: float = myopic_price(full_lambdas)

    #Dirty trick because period_profit takes a vector price, not scalar
    prices = np.array([optimal_price, optimal_price+0.01])
    return period_profit(prices, full_lambdas)[0]


price_grid = np.linspace(min_price, max_price, num=length_of_price_grid)
simplex3d = src.generate_simplex_3dims(n_per_dim=n_of_lambdas_per_dim)


if __name__ == "__main__":
    v, policy, error = src.compute_fixed_point(src.bellman_operator, v0, error_tol=error_tol,
                                               max_iter=max_iters, verbose=1,
                                               skip=1, eval_grid=simplex3d[:, 0:2],
                                               price_grid=price_grid,
                                               lambda_simplex=simplex3d,
                                               period_return_f=period_profit)

    print("Final Error : ", error)
    print("Done {0} iterations in {1} minutes".format(max_iters, (time.time() - start)/60))

    now = datetime.datetime.now()
    year, month, day = now.year, now.month, now.day

    d_ = {'valueF': v, 'policy': policy, 'error': error,
          'length_of_price_grid': length_of_price_grid,
          'n_of_lambdas_per_dim': n_of_lambdas_per_dim,
          'min_price': min_price, 'max_price': max_price}

    with open('../data/{0}-{1}-{2}vfi_dict.dill'.format(year, month, day),
              'wb') as file:
        dill.dump(d_, file)

    print("saved file: {0}-{1}-{2}vfi_dict.dill".format(year, month, day))