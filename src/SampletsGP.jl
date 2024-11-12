module SampletsGP

using PyCall
using Flux
using SparseArrays
using LinearAlgebra
using Statistics
using Printf

mutable struct GaussianProcess{T <: Real}
    X_train_normalized::Union{Nothing, Matrix{T}}           # Normalized training points
    X_train_stats::Union{Nothing, Vector{Matrix{T}}}        # Vector [X_train_mean, X_train_std], containing mean and standard deviation of X_train
    y_train_normalized::Union{Nothing, Matrix{T}}           # Normalized training values
    y_train_stats::Union{Nothing, Vector{T}}                # Vector [y_train_mean, y_train_std], containing mean and standard deviation of y_train
    kernel::String                                          # Name of the used kernel
    hyperparams::Vector{T}                                  # Hyperparameters: [lengthscale, signal variance s², noise variance σ²]
    q::Int                                                  # Amount of vanishing moments that the samplets posess
    η::T                                                    # Compression parameter when constructing the compressed kernel matrix Kₑₜₐ
    threshold::T                                            # Additional threshold for the entries in Kₑₜₐ
    ST::Union{Nothing, PyObject}                            # Samplet tree
    C::Union{Nothing, SparseArrays.CHOLMOD.Factor}          # Cholesky decomposition of Kₑₜₐ + σ²I
    alpha::Union{Nothing, Matrix{T}}                        # Solution of (Kₑₜₐ + σ²I) Tα = Ty, where y is normalized y_train
    log_marginal_likelihood::Union{Nothing, T}              # Log marginal likelihood of the trained model given hyperparams and the normalized data

    function GaussianProcess(; kernel::String = "MATERN52", hyperparams::Vector{T} = [1.0, 1.0, 1.0], q::Int = 5, η::T = 1.0, threshold::T = 10^-3) where {T <: Real}
        @assert hyperparams[1] > 0 "Lengthscale must be greater than 0."
        @assert hyperparams[2] > 0 "Signal variance must be greater than 0."
        @assert hyperparams[3] >= 0 "Noise variance must be greater than or equal to 0."
        @assert q >= 0 "Amount of vanishing moments must be at least 0."
        @assert η > 0 "Compression parameter must be greater than 0."
        @assert threshold > 0 "Threshold must be greater than 0."

        return new{T}(nothing, nothing, nothing, nothing, kernel, hyperparams, q, η, threshold, nothing, nothing, nothing, nothing)
    end
end

# Includes the SampletsBO file
include("SampletsBO.jl")

# Export functions
export GaussianProcess, train!, predict!, optimize

function train!(model::GaussianProcess, X_train::Matrix{T}, y_train::Matrix{T}, num_steps::Int = 50) where {T <: Real}
    hyperparams = model.hyperparams
    kernel = model.kernel
    q = model.q
    η = model.η
    threshold = model.threshold

    # Handle X_train as (dim x N) matrix and y_train as (1 x N) matrix
    if size(y_train)[1] > size(y_train)[2]
        X_train = X_train'
        y_train = y_train'
    end
    @assert size(X_train)[2] == size(y_train)[2] "Training points and training values must have the same amount of elements."

    # Normalize X_train
    model.X_train_stats = [mean(X_train, dims=2), std(X_train, dims=2)]
    X_train_normalized = (X_train .- model.X_train_stats[1]) ./ model.X_train_stats[2]
    model.X_train_normalized = X_train_normalized

    # Normalize y_train
    model.y_train_stats = [mean(y_train), std(y_train)]
    y_train_normalized = (y_train .- model.y_train_stats[1]) ./ model.y_train_stats[2]

    # Import the FMCA Python module (Developed in C++)
    FMCA = pyimport("FMCA")

    # Construct Samplet tree and add it to the model
    ST = FMCA.SampletTree(X_train_normalized, q)
    model.ST = ST

    # Reorder data according to cluster tree ordering and add them to the model
    indices = Int.(ST.indices()) .+ 1
    model.X_train_normalized = X_train_normalized[:, indices]
    y_train_normalized = y_train_normalized[:, indices]
    model.y_train_normalized = y_train_normalized

    # Set hyperparameter constraints
    lengthscale_constraint = [0.005, 2.0]
    signal_constraint = [0.05, 20.0]
    noise_constraint = [0.1, 2.0]

    # Define the optimizer (Adam with learning rate 0.1)
    optimizer = Flux.setup(Adam(0.1), hyperparams)

    # Training loop for optimizing the hyperparameters
    for i in 1:num_steps

        neg_log_marginal_likelihood, likelihood_gradient = negative_log_marginal_likelihood_and_gradient(X_train_normalized, y_train_normalized, hyperparams, kernel, η, threshold, ST, true)
        
        @printf("Iteration %d/%d - Loss = %.2f   lengthscale: %.4f   signal variance: %.4f   noise variance: %.4f\n", 
            i-1, num_steps, neg_log_marginal_likelihood, hyperparams[1], hyperparams[2], hyperparams[3])

        # Perform one optimization step
        Flux.update!(optimizer, hyperparams, likelihood_gradient)

        # Enforce constraints on the hyperparameters using `clamp`
        hyperparams[1] = clamp(hyperparams[1], lengthscale_constraint[1], lengthscale_constraint[2])
        hyperparams[2] = clamp(hyperparams[2], signal_constraint[1], signal_constraint[2])
        hyperparams[3] = clamp(hyperparams[3], noise_constraint[1], noise_constraint[2])
    end

    # Update trained hyperparameters
    model.hyperparams = hyperparams

    # Compute Cholesky, alpha and likelhood of the trained model to perform prediction
    C, alpha, neg_log_marginal_likelihood = negative_log_marginal_likelihood_and_gradient(X_train_normalized, y_train_normalized, hyperparams, kernel, η, threshold, ST, false)
    model.C = C
    model.alpha = alpha
    model.log_marginal_likelihood = -neg_log_marginal_likelihood

    @printf("Iteration %d/%d - Loss = %.2f   lengthscale: %.4f   signal variance: %.4f   noise variance: %.4f\n", 
        num_steps, num_steps, neg_log_marginal_likelihood, hyperparams[1], hyperparams[2], hyperparams[3])
end

function predict!(model::GaussianProcess, X_pred::Matrix{T}, return_std = false) where {T <: Real}
    lengthscale, s², σ² = model.hyperparams
    X_train_normalized = model.X_train_normalized
    X_train_mean, X_train_std = model.X_train_stats
    y_train_mean, y_train_std = model.y_train_stats
    kernel = model.kernel
    ST = model.ST
    C = model.C
    alpha = model.alpha

    # Transpose X_pred if necessary (under the assumptions N > dim)
    if size(X_pred)[1] > size(X_pred)[2]
        X_pred = X_pred'
    end

    # Apply normalization
    X_pred_normalized = (X_pred .- X_train_mean) ./ X_train_std

    # Import the FMCA Python module (Developed in C++)
    FMCA = pyimport("FMCA")

    # Compute mean prediction
    covariance = FMCA.CovarianceKernel(kernel, lengthscale)
    K₁ = s² * covariance.eval(X_pred_normalized, X_train_normalized)
    y_pred = K₁ * alpha
    y_pred = y_train_std * y_pred .+ y_train_mean           # undo normalisation

    # Compute standard deviation prediction or covariance matrix prediction
    N = length(C.p)
    K₁ = sparse(1:N, C.p, ones(N)) * FMCA.sampletTransform(ST, K₁')
    A = sparse(C.L) \ K₁
    if return_std
        std_pred = s² .- dot.(eachcol(A), eachcol(A))
        std_pred[std_pred .< 0] .= 0                        # Negative entries are set to 0
        std_pred = sqrt.(std_pred)
        std_pred = y_train_std * std_pred                   # undo normalisation

        return y_pred, std_pred
    else
        covariance_pred = s² * covariance.eval(X_pred_normalized, X_pred_normalized) - A' * A
        covariance_pred[diagind(covariance_pred)] .= max.(diag(covariance_pred), 0) # Negative diagonal entries are set to 0
        covariance_pred = y_train_std^2 * covariance_pred   # undo normalisation

        return y_pred, covariance_pred
    end
end

function negative_log_marginal_likelihood_and_gradient(X_train_normalized, y_train_normalized, hyperparams, kernel, η, threshold, ST, eval_gradient)
    # Extract hyperparameters
    lengthscale, s², σ² = hyperparams

    # Import the FMCA Python module (Developed in C++)
    FMCA = pyimport("FMCA")

    # Compute compressed kernel matrix
    covariance = FMCA.CovarianceKernel(kernel, lengthscale)
    Kₑₜₐ = FMCA.SampletKernelCompressor(ST, covariance, X_train_normalized, η, threshold).matrix()
    Kₑₜₐ = Convert_to_SparseMatrixCSC(Kₑₜₐ)                 # Convert Kₑₜₐ to SparseMatrixCSC format
    Kₑₜₐ = Kₑₜₐ + transpose(triu(Kₑₜₐ, 1))                  # Adding the transpose of the upper triangle
    Kₑₜₐ = s² * Kₑₜₐ                                        # Multiplying the signal variance

    # Compute cholesky decomposition of K̂ₑₜₐ = Kₑₜₐ + σ²I and solve the linear system (Kₑₜₐ + σ²I) Tα = Ty
    C = cholesky(Kₑₜₐ, shift = σ², check = true)
    Ty = FMCA.sampletTransform(ST, y_train_normalized')
    Talpha = C \ Ty
    alpha = FMCA.inverseSampletTransform(ST, Talpha)

    # Compute negative log marginal likelihood
    N = length(y_train_normalized)
    neg_log_marginal_likelihood = 0.5 * dot(y_train_normalized, alpha)
    neg_log_marginal_likelihood += sum(log.(diag(sparse(C.L))))
    neg_log_marginal_likelihood += 0.5 * N * log(2π)

    if eval_gradient
        # Compute derivative (∂K̂/∂ϑ)ₑₜₐ where ϑ = lengthscale
        covariance_dev = FMCA.CovarianceKernel(string(kernel, "DEV"), lengthscale)
        derivative_wrt_lengthscaleₑₜₐ = FMCA.SampletKernelCompressor(ST, covariance_dev, X_train_normalized, η, threshold).matrix()
        derivative_wrt_lengthscaleₑₜₐ = Convert_to_SparseMatrixCSC(derivative_wrt_lengthscaleₑₜₐ)
        derivative_wrt_lengthscaleₑₜₐ = derivative_wrt_lengthscaleₑₜₐ + transpose(triu(derivative_wrt_lengthscaleₑₜₐ, 1))
        if kernel == "EXPONENTIAL"
            derivative_wrt_lengthscaleₑₜₐ = s² * derivative_wrt_lengthscaleₑₜₐ / lengthscale^2
        else
            v = parse(Int, kernel[end-1])
            derivative_wrt_lengthscaleₑₜₐ = s² * sqrt(v/2) / lengthscale^2 * derivative_wrt_lengthscaleₑₜₐ  - Kₑₜₐ / lengthscale
        end

        # Compute gradient of negative log marginal likelihood
        neg_log_marginal_likelihood_derivative_wrt_lengthscale = 0
        neg_log_marginal_likelihood_derivative_wrt_s² = 0
        neg_log_marginal_likelihood_derivative_wrt_σ² = 0
        for _ in 1:50
            z = randn(N)
            u = C \ z 
            neg_log_marginal_likelihood_derivative_wrt_lengthscale += u' * (derivative_wrt_lengthscaleₑₜₐ * z)
            neg_log_marginal_likelihood_derivative_wrt_s² += u' * (Kₑₜₐ / s² * z)
            neg_log_marginal_likelihood_derivative_wrt_σ² += u' * z
        end
        neg_log_marginal_likelihood_derivative_wrt_lengthscale = - 0.5 * dot(Talpha, derivative_wrt_lengthscaleₑₜₐ * Talpha) + neg_log_marginal_likelihood_derivative_wrt_lengthscale / 100
        neg_log_marginal_likelihood_derivative_wrt_s² = - 0.5 * dot(Talpha, (Kₑₜₐ / s²) * Talpha) + neg_log_marginal_likelihood_derivative_wrt_s² / 100
        neg_log_marginal_likelihood_derivative_wrt_σ² = - 0.5 * dot(alpha, alpha) + neg_log_marginal_likelihood_derivative_wrt_σ² / 100
    
        likelihood_gradient = [neg_log_marginal_likelihood_derivative_wrt_lengthscale, neg_log_marginal_likelihood_derivative_wrt_s², neg_log_marginal_likelihood_derivative_wrt_σ²]

        return neg_log_marginal_likelihood, likelihood_gradient
    end

    return C, alpha, neg_log_marginal_likelihood
end

# Convert PyObject into sparse matrix
function Convert_to_SparseMatrixCSC(K)
    m, n = K.shape
    colPtr = Vector{Int}(PyArray(K."indptr"))
    rowVal = Vector{Int}(PyArray(K."indices"))
    nzVal = Vector{Float64}(PyArray(K."data"))
    
    @inbounds for i in eachindex(colPtr)
        colPtr[i] += 1
    end
    @inbounds for i in eachindex(rowVal)
        rowVal[i] += 1
    end
    
    return SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)
end

end