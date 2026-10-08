# ============================================================
# gw_moment.jl
#
# Run GW (Eq. 12–13) moment estimators ONLY at:
#   N = Nmax = T - 1
# for ALL supercritical test trajectories (lambda_true > mu_true)
#
# Prints:
#   MAE(lambda_hat), MAE(mu_hat)
#
# Usage:
#   julia --project=. estimators/gw_moment.jl <data_file>
# ============================================================

using JLD2
using Statistics
using DataFrames
using CSV

# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
if length(ARGS) < 1
    println("Usage: julia gw_consistency_Nmax_mae.jl <data_file>")
    exit(1)
end

data_file = ARGS[1]

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
@load data_file theta_test Z_test times m T

dt = times[2] - times[1]
K  = size(theta_test, 2)

println("Loaded: ", data_file)
println("dt = ", dt, " | T = ", T, " | m = ", m, " | K_test = ", K)

# ------------------------------------------------------------
# Helper: take replicate 1 as a single path
# ------------------------------------------------------------
get_single_path(Zk::AbstractMatrix) = Float64.(vec(Zk[:, 1]))

# ------------------------------------------------------------
# Compute m_hat and sigma2_hat from ONE trajectory Z0..ZN
# ------------------------------------------------------------
function mhat_sigma2hat(Z::AbstractVector{<:Real})
    N = length(Z) - 1
    @assert N >= 2 "Need at least N >= 2."

    Zk   = Z[1:end-1]
    Zkp1 = Z[2:end]

    # avoid divide-by-zero
    mask = Zk .> 0
    Zk   = Zk[mask]
    Zkp1 = Zkp1[mask]

    m_hat = sum(Zkp1) / sum(Zk)
    sigma2_hat = mean(((Zkp1 .- m_hat .* Zk) .^ 2) ./ Zk)

    return m_hat, sigma2_hat
end

# ------------------------------------------------------------
# Eq (12)-(13) GW estimators
# ------------------------------------------------------------
function gw_lambda_mu(Z::AbstractVector{<:Real}, dt::Real)
    m_hat, sigma2_hat = mhat_sigma2hat(Z)

    if !(m_hat > 0) || !isfinite(m_hat) || abs(m_hat - 1) < 1e-12
        return (lambda_hat = NaN, mu_hat = NaN, m_hat = m_hat, sigma2_hat = sigma2_hat)
    end

    frac = sigma2_hat / (m_hat * (m_hat - 1))
    pref = log(m_hat) / (2 * dt)

    lambda_hat = pref * (frac + 1)
    mu_hat     = pref * (frac - 1)

    return (lambda_hat = lambda_hat, mu_hat = mu_hat, m_hat = m_hat, sigma2_hat = sigma2_hat)
end

# ------------------------------------------------------------
# Select ALL supercritical indices (lambda_true > mu_true)
# ------------------------------------------------------------
lambda_true = vec(theta_test[1, :])
mu_true     = vec(theta_test[2, :])

super_idx = findall(i -> lambda_true[i] > mu_true[i], 1:K)
isempty(super_idx) && error("No supercritical (lambda_true > mu_true) found in theta_test.")

println("Supercritical trajectories: ", length(super_idx), " / ", K)

# ------------------------------------------------------------
# Only evaluate at Nmax
# ------------------------------------------------------------
Nmax = T - 1
println("Using ONLY Nmax = ", Nmax)

rows = DataFrame(
    k_idx       = Int[],
    lambda_true = Float64[],
    mu_true     = Float64[],
    lambda_hat  = Float64[],
    mu_hat      = Float64[],
)

for k in super_idx
    Zfull = get_single_path(Z_test[k])
    Z = Zfull[1:(Nmax + 1)]  # full length

    est = gw_lambda_mu(Z, dt)

    push!(rows, (
        k,
        lambda_true[k],
        mu_true[k],
        est.lambda_hat,
        est.mu_hat
    ))
end

# ------------------------------------------------------------
# Compute MAE (ignore NaNs)
# ------------------------------------------------------------
ok_lam = isfinite.(rows.lambda_hat)
ok_mu  = isfinite.(rows.mu_hat)

mae_lambda = mean(abs.(rows.lambda_hat[ok_lam] .- rows.lambda_true[ok_lam]))
mae_mu     = mean(abs.(rows.mu_hat[ok_mu]     .- rows.mu_true[ok_mu]))

println("\n========== GW @ Nmax TEST PERFORMANCE ==========")
println("Nmax = ", Nmax, "  (t_end = ", times[Nmax+1], ")")
println("Valid lambda estimates: ", sum(ok_lam), " / ", nrow(rows))
println("Valid mu estimates:     ", sum(ok_mu),  " / ", nrow(rows))
println("MAE(lambda_hat) = ", round(mae_lambda, digits=6))
println("MAE(mu_hat)     = ", round(mae_mu, digits=6))
println("==============================================\n")

# Optional: save per-trajectory results
CSV.write("gw_Nmax_results.csv", rows)
println("Saved: gw_Nmax_results.csv")