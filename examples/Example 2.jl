using SampletsGP, Random, LinearAlgebra

# Set the random seed
Random.seed!(0)

# Generate random points uniformly in the interval [-5.12, 5.12]
dim = 2
N = 100000
m = 1000
X_train = 10.24 * rand(dim, N) .- 5.12
X_test = 10.24 * rand(dim, m) .- 5.12

# Calculate y_train and y_test based on the Rastrigin function
y_train = 10 * dim .+ sum(X_train.^2 .- 10 * cos.(2 .* π .* X_train), dims=1) + randn(N)'
y_test = 10 * dim .+ sum(X_test.^2 .- 10 * cos.(2 .* π .* X_test), dims=1)

# Define and train the GP model
model = GaussianProcess(q = 5, threshold = 10^-3, hyperparams = [1.0, 1.0, 1.0])
train!(model, X_train, y_train, 2)

# Make predictions using the GP model
y_pred, cov_pred = predict!(model, X_test)

# Compute relative error
println("Relative error is: ", norm(y_pred' - y_test)/norm(y_test))