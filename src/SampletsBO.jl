function optimize(f::Function, lower_bound::Vector, upper_bound::Vector, N₀::Int, γ::Real, model::GaussianProcess = GaussianProcess())
    @assert N₀ > 100 "Amount of optimization steps must be bigger than 100."
    @assert 0 < γ ≤ 1 "Gamma must be in the interval (0, 1]."

    # Start with the first 99 observation, randomly generated in the bounded region
    dim = length(lower_bound)
    X_train = (upper_bound - lower_bound) .* rand(dim, 99) .+ lower_bound
    y_train = reshape([f(X_train[:, i]) for i in 1:99], 1, 99)

    for i in 100:(N₀-1)
        # Train GP if i is a multiple of 100
        if i % 100 == 0
            train!(model, X_train, y_train, 2)
        end

        # Generate random points uniformly in the bounded region
        X = (upper_bound - lower_bound) .* rand(dim, 100*dim) .+ lower_bound

        # Make predictions, compute the posterior covariance and perform Cholesky decomposition
        y_pred, covariance_pred = predict!(model, X)
        L = cholesky(Hermitian(covariance_pred), NoPivot()).L
        
        # Generate noise and compute the function values of the Thompson sample
        ε =  randn(100*dim)
        a = y_pred + L * ε

        # Find the indices where var_pred is greater than or equal to the maximum value in var_pred times γ
        var_pred = diag(covariance_pred)
        indices = findall(i -> var_pred[i] >= γ * maximum(var_pred), eachindex(var_pred))
        a = a[indices]
        X = X[:, indices]

        # Get the next point to sample
        x_next = X[:, argmax(a)]
        y_next = f(x_next)

        # Append the point to the training set
        X_train = hcat(X_train, x_next)
        y_train = hcat(y_train, y_next)
    end

    # Train GP
    train!(model, X_train, y_train, 2)

    # Generate random points uniformly in the bounded region
    X = (upper_bound - lower_bound) .* rand(dim, 100*dim) .+ lower_bound

    # Make predictions, compute the posterior covariance
    y_pred, std_pred = predict!(model, X, true)

    # Select the point that maximizes the predicted function values (y_pred)
    x_next = X[:, argmax(y_pred)]
    y_next = f(x_next)

    # Add this point to the training set
    X_train = hcat(X_train, x_next)
    y_train = hcat(y_train, y_next)

    # Find the overall optimum
    x_optimum, y_optimum = X_train[:, argmax(y_train)[2]], maximum(y_train)

    return x_optimum, y_optimum
end