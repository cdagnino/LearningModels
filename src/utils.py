import time
import numpy as np


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
