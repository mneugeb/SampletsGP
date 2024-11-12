using SampletsGP, Random, Plots

# Set the random seed
Random.seed!(0)

# Define data
N = 20000
X_train = reshape(collect(range(-0.9, 0.9, length=N)), N, 1)
X_pred = reshape(collect(range(-0.9, 0.9, length=100)), 100, 1)
y_train = 2 .* X_train .* sin.(4 .* X_train.^2) .* cos.(4 .* X_train.^3) + randn(N)

# Define GaussianProcess model. This is simply the model GaussianProcess(kernel = "MATERN52", hyperparams = [1.0, 1.0, 1.0], q = 5, η = 1.0, threshold = 10^-3)
model = GaussianProcess()

# Train model
train!(model, X_train, y_train, 5)

# Make predictions using the GP model
y_pred, std_pred = predict!(model, X_pred, true)

# If the full covariance matrix is of interest, use the next two lines
# y_pred, cov_pred = predict!(model, X_pred)
# std_pred = sqrt.(diag(cov_pred))

# Define the true function (without noise)
f(x) = 2 .* x .* sin.(4 .* x.^2) .* cos.(4 .* x.^3)

# Plot the confidence intervals, true function, and posterior GP mean
plot(X_pred, y_pred .- 1.96 .* std_pred, fillrange = y_pred .+ 1.96 .* std_pred, fillalpha=0.5, label="95% confidence interval", lw=0, color=:cyan, ylims=(-1, 1))
plot!(X_pred, f(X_pred), label="True Function", color="red", lw = 3, title="Gaussian Process Regression via Samplets", xlabel="x", ylabel="y", legendfont=font(9), tickfont=font(9), grid=false)
plot!(X_pred, y_pred, label="Posterior GP Mean", color="blue", lw = 3)
annotate!((0.8, 0.9, text("N=$N", :black, 12)))