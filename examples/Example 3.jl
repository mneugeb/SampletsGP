using SampletsGP

# Define negative Ackley function
function negativeAckley(x)
    x1, x2 = x[1], x[2]
    c = 2π

    return 20 * exp(-0.2 * sqrt(0.5 * (x1^2 + x2^2))) + exp(0.5 * (cos(c * x1) + cos(c * x2))) - exp(1) - 20
end

# Define bounds for the domain of the negative Ackley function
lower_bound = [-5, -5]
upper_bound = [5, 5]

# Define SampletsGP model
model = GaussianProcess()

# Amount of points to optimize the negative Ackley function
N₀ = 1000

# Optimize negative Ackley function
x_optimum, y_optimum = optimize(negativeAckley, lower_bound, upper_bound, N₀, 0.5, model)

println("After ", N₀, " optimization steps, the optimal solution was found with x = ", x_optimum, " and y = ", y_optimum)