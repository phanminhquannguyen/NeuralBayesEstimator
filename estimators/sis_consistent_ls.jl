# ============================================================
# This file implements a consistenct least square estimator for the SIS (Verhulst) model, 
# based on the paper "Consistent Least Squares Estimation in PSDBP" by Peter Braunstein, Sophie Hauntphenee, and Carmen Minuesa
#
# For each trajectory i:
#   - estimate (gamma, nu) by minimizing weighted LS objective
#   - compare to true (gamma, nu)
#
# Output of the file is:
#   - trajectory length (number of observations)
#   - MAE(gamma), MAE(nu) across trajectories
#
# Command to run this file: julia --project=. estimators/sis_consistent_ls.jl
# ============================================================

using JLD2
using LinearAlgebra
using Statistics
using Optim
using Printf

# Settings for the estimator
data_file   = "data/sis_non_extinct.jld2"
split       = :test          
max_iters   = 200
theta0      = [1.0, 1.0]     # initial guess [gamma, nu] => subject to change


# Load simulated dataset
@load data_file Z_train Z_val Z_test theta_train theta_val theta_test dt N

# Ensure the structure of the simulated data is corrected
Z_list =
    split == :train ? Z_train :
    split == :val   ? Z_val   :
    split == :test  ? Z_test  :
    error("split must be :train, :val, or :test")

theta_true =
    split == :train ? theta_train :
    split == :val   ? theta_val   :
    split == :test  ? theta_test  :
    error("split must be :train, :val, or :test")

K = length(Z_list)
K_use = K

traj_length = size(Z_list[1], 1)  # number of observations per trajectory (rows)

println("dt = $dt | N = $N")
println("Trajectory length (num observations) = $traj_length")
println("Using K_use = $K_use out of K = $K\n")

# SIS generator
# lambda_z = gamma * z * (1 - z/N)
# mu_z     = nu * z
function build_generator(gamma, nu, N)
    S = N + 1
    G = zeros(Float64, S, S)

    for z in 0:N
        i = z + 1
        lambda = gamma * z * (1 - z / N)
        mu     = nu * z

        if z < N
            G[i, i+1] = lambda
        end
        if z > 0
            G[i, i-1] = mu
        end

        G[i, i] = -(lambda + mu)
    end

    return G
end


# Q-process mean m_up(z, theta)
function qprocess_mean(gamma, nu, N, dt)
    G = build_generator(gamma, nu, N)
    P = exp(dt * G)          # one-step transition matrix
    Q = P[2:end, 2:end]      # remove extinction state

    E = eigen(Matrix(Q))
    idx = argmax(real.(E.values))

    rho = real(E.values[idx])
    v   = abs.(real.(E.vectors[:, idx])) .+ 1e-12

    m_up = zeros(Float64, N)
    for z in 1:N
        denom = rho * v[z]
        s = 0.0
        for j in 1:N
            s += j * Q[z, j] * v[j] / denom
        end
        m_up[z] = s / z
    end

    return m_up
end

# Build objective for a SINGLE trajectory Z (captures Z in closure)
function make_objective(Z::Vector{Int}, N::Int, dt::Float64)
    n = length(Z)

    return function objective(theta)
        gamma = theta[1]
        nu    = theta[2]

        if !(isfinite(gamma) && isfinite(nu)) || gamma <= 0 || nu <= 0
            return Inf
        end

        m_up = qprocess_mean(gamma, nu, N, dt)

        sse = 0.0
        for k in 2:n
            z_prev = Z[k-1]
            z_now  = Z[k]
            if z_prev == 0
                continue
            end

            predicted = z_prev * m_up[z_prev]
            err       = z_now - predicted
            weight    = 1.0 / (z_prev^2)

            sse += weight * (err^2)
        end

        return sse
    end
end

# Fit one trajectory -> theta_hat
function fit_one_trajectory(Zmat::Matrix{Float64}, theta0, N, dt; max_iters=200)
    Z = Int.(round.(Zmat[:, 1]))  # use first replicate col

    obj = make_objective(Z, N, dt)

    opts = Optim.Options(iterations=max_iters)
    res  = Optim.optimize(obj, theta0, NelderMead(), opts)

    theta_hat = Optim.minimizer(res)
    return theta_hat
end


# Loop over trajectories and compute MAE
gamma_errs = Float64[]
nu_errs    = Float64[]

for i in 1:K_use
    Zmat = Float64.(Z_list[i])
    theta_hat = fit_one_trajectory(Zmat, theta0, N, dt; max_iters=max_iters)

    gamma_hat = theta_hat[1]
    nu_hat    = theta_hat[2]

    gamma_true = theta_true[1, i]
    nu_true    = theta_true[2, i]

    push!(gamma_errs, abs(gamma_hat - gamma_true))
    push!(nu_errs,    abs(nu_hat - nu_true))
end

println("\n========== FINAL RESULTS ==========")
println("K_used = ", K_use)
println("Trajectory length (num observations) = ", traj_length)
println("MAE(gamma) = ", mean(gamma_errs))
println("MAE(nu)    = ", mean(nu_errs))
println("===================================\n")