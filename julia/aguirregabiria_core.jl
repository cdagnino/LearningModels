#
#
#

using GridInterpolations
using Distributions
using QuadGK

betas_transition = [-3.0, -2.5, -2.0]
sigma_eps = 0.5
alpha = 1.0
c = 0.5
delta = 0.9

function generate_simplex_3dims(n_per_dim::Integer=20) 

    simplex = [[x, y, 1-x-y] for x in linspace(0,1,n_per_dim) for y in linspace(0,1,n_per_dim) if x+y <= 1.0]

    return hcat(simplex...)'

end

function interpV(simplex, V)

    step = simplex[2,2]
    grid = 0.0:step:1.0
    size_simplex = length(grid)

    augment_V = zeros(size_simplex, size_simplex)
    k = 1
    for i in 1:size_simplex
        for j in 1:size_simplex
            if j <= size_simplex+1-i
                augment_V[i,j] = V[k]
                k += 1
            end
        end
    end

    interpolate_V(x) = interpolate(SimplexGrid(grid, grid), flipdim(augment_V,1), [x(1), 1-x(2)])

    return interpolate_V

end

#println(generate_simplex_3dims(5))

function dmd_transition_fs(new_state, action, old_state)
    return [pdf(Normal(alpha + beta*log(action), sigma_eps), x) for x in betas_transition]
end

function belief(new_state, action, old_state, lambda_weights)
    return dot(dmd_transition_fs(new_state, action, old_state), lambda_weights)
end 

function update_lambdas(new_state, action, old_state, old_lambdas)
    den = sum(old_lambdas * dmd_transition_fs(new_state, action, old_state))
    return dmd_transition_fs(new_state, action, old_state)*old_lambdas / den
end

function period_return(price_grid, lambdas)
    # Get E[β] from the lambdas
    E_beta = dot(lambdas, betas_transition)
    logq = alpha + E_beta * log.(price_grid)

    return (price_grid - c) * exp(logq)

end

function E0fV(Vguess, price_grid, lambda_weights)
    # Integrates Vguess * belief_f for each value of p. Integration over demand
    # Sum of points on demand and weights, multiplied by V and the belief function

    integrated_values = similar(price_grid)

    for i in 1:length(integrated_values)

        new_lambdas(x) = update_lambdas(x, price_grid[i], ~, lambda_weights)
        new_belief(x)  = belief(x, price_grid[i], ~, lambda_weights)
        
        #Vguess takes all lambdas except last (because of the simplex)
        integrand(x) = Vguess[new_lambdas(x)[1:end-1]] * new_belief(x)

        logd_min, logd_max = -6, 2.3 #D = (0.01, 10)
        integrated_values[i] = quadgk(integrand, logd_min, logd_max, maxevals=250)

    end

    return integrated_values

end

function bellman_operator(Vguess, price_grid, lambda_simplex)

    policy = similar(lambda_simplex)
    T_V = similar(lambda_simplex)

    # 1. Go over grid of state space
    # 2. Write objective (present return + delta*eOfV)
    # 3. Find optimal p on that objective
    # 4. Write optimal p and value function on that point in the grid

    for index in size(lambda_simplex, 1)

        println("> Doing ", index, " of ", length(lambda_simplex))

        lambda_weights = lambda_simplex[index, :]
        
        R      = period_return(price_grid, lambda_weights)
        E0fV_p = E0fV(Vguess, price_grid, lambda_weights)

        objective_vals = R + delta *  E0fV_p

        ind_max, T_V[index] = findmax(objective_vals)
        policy[index]       = price_grid(ind_max)

    end

    # Interpolate using Interpolations.jl
    interp_T_V = interpV(lambda_simplex, T_V)

    return interp_T_V, policy

end

function compute_fixed_point(V, price_grid, lambda_simplex, error_tol=1e-5, max_iter=50, verbose=1, skip=10)

    iterate = 0
    error = error_tol + 1

    while iterate < max_iter && error > error_tol
        start_iter = tic()

        new_V, policy = bellman_operator(V, price_grid, lambda_simplex)

        iterate += 1
        error = max(abs.(new_V - V))

        V = new_V

        if verbose && mod(iterate, skip) == 0
            time_taken = (toc() - start_iter) / 60.0
            println("Computed iterate %d with error %f in %f minutes", iterate, error, time_taken)
        end
    end

    return V, policy, error

end
