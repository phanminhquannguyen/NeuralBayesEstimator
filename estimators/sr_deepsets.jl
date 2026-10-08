# ============================================================
# sr_deepsets.jl
#
# Two-network DeepSets (works with m = 1):
#   R-network: r = λ - μ
#   S-network: log_s = log(λ + μ)
#
# Usage:
#   julia --project=. estimators/sr_deepsets.jl <extinct|non_extinct|non_extinct_Q>
# ============================================================

using Random
using JLD2
using Flux
using NeuralEstimators
using BSON
using Statistics

# ------------------------------------------------------------
# 0) Robust preprocessing: clip(log1p(Z)) + GLOBAL robust scale
# ------------------------------------------------------------

function fit_global_robust_normalizer(Z_train::Vector{Matrix{Float64}};
                                      qlo::Float64=0.001,
                                      qhi::Float64=0.999,
                                      eps::Float64=1e-6)

    vals = Float64[]
    sizehint!(vals, sum(length, Z_train))
    for Zk in Z_train
        append!(vals, vec(log1p.(Zk)))
    end

    lo = quantile(vals, qlo)
    hi = quantile(vals, qhi)

    vals_clip = clamp.(vals, lo, hi)
    med = median(vals_clip)
    mad = median(abs.(vals_clip .- med)) + eps

    return (lo=lo, hi=hi, med=med, mad=mad, eps=eps, qlo=qlo, qhi=qhi)
end

function preprocess_log_clip_robust(Zk::AbstractMatrix{<:Real}, norm)
    X = log1p.(Float64.(Zk))
    X = clamp.(X, norm.lo, norm.hi)
    return (X .- norm.med) ./ norm.mad
end

preprocess_dataset(Z_list, norm) = [preprocess_log_clip_robust(Zk, norm) for Zk in Z_list]

# ------------------------------------------------------------
# 1) Targets
# ------------------------------------------------------------

to_r(theta_lm::AbstractMatrix{<:Real}) =
    reshape(theta_lm[1, :] .- theta_lm[2, :], 1, :)

function to_log_s(theta_lm::AbstractMatrix{<:Real}; eps::Real=1e-8)
    s = theta_lm[1, :] .+ theta_lm[2, :]
    return reshape(log.(Float64.(s) .+ eps), 1, :)
end

# ------------------------------------------------------------
# 2) Stage-wise stats for S-network
# ------------------------------------------------------------

function augment_with_stage_stats_levels_and_increments(Zk_normed::AbstractMatrix{<:Real}; nstages::Int=10)
    T, m = size(Zk_normed)

    cuts = [max(2, Int(ceil(j * T / nstages))) for j in 1:nstages]
    cuts[end] = T

    dZ = diff(Zk_normed; dims=1)  # (T-1) x m

    lvl_mean = Matrix{Float64}(undef, nstages, m)
    lvl_var  = Matrix{Float64}(undef, nstages, m)

    inc_mean = Matrix{Float64}(undef, nstages, m)
    inc_var  = Matrix{Float64}(undef, nstages, m)
    inc_mad  = Matrix{Float64}(undef, nstages, m)

    for (j, tj) in enumerate(cuts)
        Xlvl = @view Zk_normed[1:tj, :]
        lvl_mean[j, :] .= vec(mean(Xlvl; dims=1))
        lvl_var[j,  :] .= vec(var(Xlvl;  dims=1))

        tdi = max(1, tj - 1)
        Xinc = @view dZ[1:tdi, :]
        inc_mean[j, :] .= vec(mean(Xinc; dims=1))
        inc_var[j,  :] .= vec(var(Xinc;  dims=1))
        inc_mad[j,  :] .= vec(mean(abs.(Xinc); dims=1))
    end

    total_abs_change = vec(sum(abs.(dZ); dims=1))
    drift            = vec(Zk_normed[end, :] .- Zk_normed[1, :])
    extra = vcat(total_abs_change', drift')  # 2 x m

    stats = vcat(lvl_mean, lvl_var, inc_mean, inc_var, inc_mad, extra)
    return vcat(Matrix{Float64}(Zk_normed), Matrix{Float64}(stats))
end

function build_augmented_dataset_sum(Z_list::Vector{Matrix{Float64}}, norm; nstages::Int=10)
    out = Vector{Matrix{Float64}}(undef, length(Z_list))
    for k in eachindex(Z_list)
        Zk_normed = preprocess_log_clip_robust(Z_list[k], norm)
        out[k] = augment_with_stage_stats_levels_and_increments(Zk_normed; nstages=nstages)
    end
    return out
end

# ------------------------------------------------------------
# 3) Loss for log_s network: SmoothL1/Huber on log_s
# ------------------------------------------------------------

function smoothl1_loss(e::AbstractArray{<:Real}; delta::Real=1.0)
    a = abs.(e)
    return mean(ifelse.(a .<= delta, 0.5 .* (a .^ 2) ./ delta, a .- 0.5*delta))
end

loss_huber_log_s(yhat, ytrue; delta::Real=1.0) = smoothl1_loss(yhat .- ytrue; delta=delta)

# ------------------------------------------------------------
# 4) L2 penalty (SAFE): collect trainables ONCE, never Flux.params in loss
# ------------------------------------------------------------

l2_penalty(trainables; weight::Float64=1e-6) =
    weight * sum(p -> sum(abs2, p), trainables)

function train_with_l2(estimator, trainables, theta_train, theta_val, Z_train, Z_val;
                       epochs::Int=40,
                       batchsize::Int=1024,
                       data_loss,
                       lr::Float64=1e-3,
                       weight_decay::Float64=1e-6,
                       use_gpu::Bool=false,
                       verbose::Bool=true)

    opt = Flux.setup(Adam(lr), estimator)
    loss_fn = (yhat, ytrue) -> data_loss(yhat, ytrue) + l2_penalty(trainables; weight=weight_decay)

    return train(
        estimator,
        theta_train,
        theta_val,
        Z_train,
        Z_val;
        epochs=epochs,
        batchsize=batchsize,
        loss=loss_fn,
        optimiser=opt,
        use_gpu=use_gpu,
        verbose=verbose
    )
end

# ------------------------------------------------------------
# 5) CLI + files
# ------------------------------------------------------------

if length(ARGS) < 1 || !(ARGS[1] in ["extinct","non_extinct","non_extinct_Q"])
    error("Usage: julia SR_network.jl <extinct|non_extinct|non_extinct_Q>")
end
case = ARGS[1]

data_file =
    case == "extinct"     ? "data/lbdp_extinct.jld2" :
    case == "non_extinct" ? "data/lbdp_non_extinct.jld2" :
                            "data/lbdp_q_subcritical_non_extinct.jld2"

model_r_file =
    case == "extinct"     ? "trained_estimator_extinct_diff_SR.jld2" :
    case == "non_extinct" ? "trained_estimator_non_extinct_diff_SR.jld2" :
                            "trained_estimator_non_extinct_Q_diff_SR.jld2"

model_s_file =
    case == "extinct"     ? "trained_estimator_extinct_logsum_SR.jld2" :
    case == "non_extinct" ? "trained_estimator_non_extinct_logsum_SR.jld2" :
                            "trained_estimator_non_extinct_Q_logsum_SR.jld2"

# ------------------------------------------------------------
# 6) Load train/val
# ------------------------------------------------------------

@load data_file theta_train theta_val Z_train Z_val times m T

theta_train_lm = Float64.(theta_train)
theta_val_lm   = Float64.(theta_val)

Z_train_raw = [Float64.(Zk) for Zk in Z_train]
Z_val_raw   = [Float64.(Zk) for Zk in Z_val]

println("Loaded: ", data_file)
println("T = ", T, " | m = ", m, " (OK if m=1)")
println("Training two networks: r=lambda-mu and log_s=log(lambda+mu)")

# ------------------------------------------------------------
# 7) Fit robust normalizer
# ------------------------------------------------------------

norm = fit_global_robust_normalizer(Z_train_raw; qlo=0.001, qhi=0.999)

# ------------------------------------------------------------
# 8) Build datasets
# ------------------------------------------------------------

# R-network inputs
theta_train_r = to_r(theta_train_lm)
theta_val_r   = to_r(theta_val_lm)

Z_train_r = preprocess_dataset(Z_train_raw, norm)
Z_val_r   = preprocess_dataset(Z_val_raw, norm)

# S-network inputs
eps = 1e-8
theta_train_log_s = to_log_s(theta_train_lm; eps=eps)
theta_val_log_s   = to_log_s(theta_val_lm; eps=eps)

nstages = 10
Z_train_s = build_augmented_dataset_sum(Z_train_raw, norm; nstages=nstages)
Z_val_s   = build_augmented_dataset_sum(Z_val_raw,   norm; nstages=nstages)

n_in_r = size(Z_train_r[1], 1)
n_in_s = size(Z_train_s[1], 1)

println("Input length r-network: ", n_in_r)
println("Input length s-network: ", n_in_s, " (nstages=", nstages, ")")

# ------------------------------------------------------------
# 9) Architectures
# ------------------------------------------------------------

w = 64

psi_r = Chain(Dense(n_in_r, w, relu), Dense(w, w, relu), Dense(w, w, relu))
phi_r = Chain(Dense(w, w, relu), Dense(w, w, relu), Dense(w, 1))
estimator_r = PointEstimator(DeepSet(psi_r, phi_r))

psi_s = Chain(Dense(n_in_s, w, relu), Dense(w, w, relu), Dense(w, w, relu))
phi_s = Chain(Dense(w, w, relu), Dense(w, w, relu), Dense(w, 1))
estimator_s = PointEstimator(DeepSet(psi_s, phi_s))

# collect trainables ONCE (critical!)
trainables_r = Flux.trainables(estimator_r)
trainables_s = Flux.trainables(estimator_s)

# ------------------------------------------------------------
# 10) Train R network
# ------------------------------------------------------------

println("\n--- Training r-network (lambda - mu) ---")
estimator_r = train_with_l2(
    estimator_r, trainables_r,
    theta_train_r, theta_val_r,
    Z_train_r, Z_val_r;
    epochs=40,
    batchsize=1024,
    data_loss=Flux.mae,
    lr=1e-3,
    weight_decay=1e-6,
    use_gpu=false,
    verbose=true
)

@save model_r_file estimator_r norm T m case n_in_r
println("Saved r-network to: ", model_r_file)

# ------------------------------------------------------------
# 11) Train S network
# ------------------------------------------------------------

println("\n--- Training s-network (log(lambda + mu)) ---")
loss_s_data = (yhat, ytrue) -> loss_huber_log_s(yhat, ytrue; delta=1.0)

estimator_s = train_with_l2(
    estimator_s, trainables_s,
    theta_train_log_s, theta_val_log_s,
    Z_train_s, Z_val_s;
    epochs=40,
    batchsize=1024,
    data_loss=loss_s_data,
    lr=1e-3,
    weight_decay=1e-6,
    use_gpu=false,
    verbose=true
)

@save model_s_file estimator_s norm T m case n_in_s nstages
println("Saved s-network to: ", model_s_file)

# ------------------------------------------------------------
# 12) Evaluate BOTH on TEST and reconstruct lambda/mu
# ------------------------------------------------------------

println("\nEvaluating on TEST data...")

@load data_file theta_test Z_test
theta_test_lm = Float64.(theta_test)

lambda_true = theta_test_lm[1, :]
mu_true     = theta_test_lm[2, :]

r_true     = lambda_true .- mu_true
s_true     = lambda_true .+ mu_true
log_s_true = log.(s_true .+ eps)

Z_test_raw = [Float64.(Zk) for Zk in Z_test]
Z_test_r = preprocess_dataset(Z_test_raw, norm)
Z_test_s = build_augmented_dataset_sum(Z_test_raw, norm; nstages=nstages)

size(Z_test_r[1], 1) == n_in_r || error("Test input mismatch r-network: expected $n_in_r got $(size(Z_test_r[1],1))")
size(Z_test_s[1], 1) == n_in_s || error("Test input mismatch s-network: expected $n_in_s got $(size(Z_test_s[1],1))")

r_hat     = vec(estimate(estimator_r, Z_test_r; use_gpu=false))
log_s_hat = vec(estimate(estimator_s, Z_test_s; use_gpu=false))

s_hat = exp.(log_s_hat)

lambda_hat = (s_hat .+ r_hat) ./ 2
mu_hat     = (s_hat .- r_hat) ./ 2

mae_r     = mean(abs.(r_hat .- r_true))
mae_s     = mean(abs.(s_hat .- s_true))
mae_log_s = mean(abs.(log_s_hat .- log_s_true))

mae_lambda = mean(abs.(lambda_hat .- lambda_true))
mae_mu     = mean(abs.(mu_hat     .- mu_true))

println("\n========== TEST PERFORMANCE ==========")
println("MAE(r = lambda - mu)        = ", round(mae_r, digits=6))
println("MAE(s = lambda + mu)        = ", round(mae_s, digits=6))
println("MAE(log(s))                 = ", round(mae_log_s, digits=6))
println("MAE(lambda)                 = ", round(mae_lambda, digits=6))
println("MAE(mu)                     = ", round(mae_mu, digits=6))
println("=====================================\n")