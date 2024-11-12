# SampletsGP

SampletsGP leverages Samplets to compute Gaussian Processes for large datasets in low-dimensional spaces. The algorithms were developed as part of my Master's thesis at the University of Cologne.

# Installation

These algorithms are designed for Linux-based environments, as they depend on the FMCA library, which is configured for such systems. On a Windows machine, a Linux-like environment can be set up using a virtual machine. You can add this package by entering the following code in the ```julia``` terminal:

```julia
using Pkg
Pkg.add(url="https://github.com/mneugeb/SampletsGP")
```
The FMCA library is designed for use in Python. To interface with it in Julia, we rely on the ```PyCall``` package. Additionally, the ```FMCA.so``` file must be accessible to Python through PyCall. To achieve this, users should place the  ```FMCA.so``` file in a directory where Python can locate it. 

The provided ```FMCA.so``` build includes kernel functions not present in the original library. This is due to the original library’s lack of support for custom kernels and its limited set, which omits some necessary ones. Therefore, the provided ```FMCA.so``` file is essential, and the code will not work with any other build.

# Example
Here is a short Example:
```julia
using SampletsGP, Plots, Random

# Define data
N = 20000
X_train = reshape(collect(range(-0.9, 0.9, length=N)), N, 1)
X_pred = reshape(collect(range(-0.9, 0.9, length=100)), 100, 1)
y_train = 2 .* X_train .* sin.(4 .* X_train.^2) .* cos.(4 .* X_train.^3) + randn(N)

# Define a GaussianProcess model
model = GaussianProcess()

# Train the model by optimizing the kernel hyperparameters five times
train!(model, X_train, y_train, 5)

# Make predictions using the GP model
y_pred, std_pred = predict!(model, X_pred, true)

# Define the true function (without noise)
f(x) = 2 .* x .* sin.(4 .* x.^2) .* cos.(4 .* x.^3)

# Plot the confidence intervals, true function, and posterior GP mean
plot(X_pred, y_pred .- 1.96 .* std_pred, fillrange = y_pred .+ 1.96 .* std_pred, fillalpha=0.5, label="95% confidence interval", lw=0, color=:cyan, ylims=(-1, 1))
plot!(X_pred, f(X_pred), label="True Function", color="red", lw = 3, title="Gaussian Process Regression via Samplets", xlabel="x", ylabel="y", legendfont=font(9), tickfont=font(9))
plot!(X_pred, y_pred, label="Posterior GP Mean", color="blue", lw = 3)
annotate!((0.8, 0.9, text("N=$N", :black, 12)))
```

# Python Requirements
This project requires:
* Python 3.12.3
* Numpy 2.1.3
* Scipy 1.14.1
* FMCA module

# References
* https://edoc.unibas.ch/87915/1/2022-05.pdf
* https://github.com/muchip/fmca/tree/master
