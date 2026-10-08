# ============================================================
# nbe_cnn.jl
# 1D-CNN baseline for inferring (lambda, mu) (or (gamma, nu) for SIS) from
# population trajectories.
#
# Usage:
#   julia --project=. estimators/nbe_cnn.jl <extinct|non_extinct|non_extinct_Q|non_extinct_SIS|extinct_SIS>
#
# Loads:  data/*.jld2 produced by the matching script in simulators/
# Prints: TEST MAE for the two rate parameters
# ============================================================

using Random
using JLD2
using Flux
using Statistics

# -------------------------
# Preprocess: log1p + global standardisation (fit on train only)
# -------------------------
function fit_global_norm(Z_train::Vector{Matrix{Float64}}; eps::Float64=1e-6)
    all_vals = Float64[]
    sizehint!(all_vals, sum(length, Z_train))
    for Zk in Z_train
        append!(all_vals, vec(log1p.(Zk)))
    end
    return (mu = mean(all_vals), sigma = std(all_vals), eps = eps)
end

function apply_global_norm!(Z_list::Vector{Matrix{Float64}}, norm)
    mu, sigma, eps = norm.mu, norm.sigma, norm.eps
    for k in eachindex(Z_list)
        X = log1p.(Z_list[k])
        Z_list[k] .= (X .- mu) ./ (sigma + eps)
    end
    return Z_list
end

# -------------------------
# Helper: T×m matrix -> (T, m, 1) tensor for Conv1D
# Flux Conv expects (length, channels, batch)
# -------------------------
to_cnn_input(Zk::AbstractMatrix{<:Real}) =
    reshape(Float32.(Zk), size(Zk, 1), size(Zk, 2), 1)

# -------------------------
# CLI
# -------------------------
if length(ARGS) < 1 || !(ARGS[1] in ["extinct", "non_extinct", "non_extinct_Q", "non_extinct_SIS", "extinct_SIS"])
    println("Usage: julia NBE_training.jl <extinct|non_extinct|non_extinct_Q|non_extinct_SIS|extinct_SIS>")
    exit(1)
end
case = ARGS[1]

data_file =
    case == "extinct"          ? "data/lbdp_extinct.jld2" :
    case == "non_extinct"      ? "data/lbdp_non_extinct.jld2" :
    case == "non_extinct_Q"    ? "data/lbdp_q_subcritical_non_extinct.jld2" :
    case == "non_extinct_SIS"  ? "data/sis_non_extinct.jld2" :
    case == "extinct_SIS"      ? "data/sis_extinct.jld2" :
                                 error("Unknown case: $case")

# -------------------------
# Load data (train/val/test)
# -------------------------
@load data_file theta_train theta_val theta_test Z_train Z_val Z_test times m T

theta_train64 = Float64.(theta_train)
theta_val64   = Float64.(theta_val)
theta_test64  = Float64.(theta_test)

Z_train64 = [Float64.(Zk) for Zk in Z_train]
Z_val64   = [Float64.(Zk) for Zk in Z_val]
Z_test64  = [Float64.(Zk) for Zk in Z_test]

println("Loaded: ", data_file)
println("T=", T, " | m=", m,
        " | K_train=", size(theta_train64,2),
        " | K_val=", size(theta_val64,2),
        " | K_test=", size(theta_test64,2))

# -------------------------
# Preprocess
# -------------------------
norm = fit_global_norm(Z_train64)
apply_global_norm!(Z_train64, norm)
apply_global_norm!(Z_val64,   norm)
apply_global_norm!(Z_test64,  norm)

# -------------------------
# CNN model
# Input: (T, m, batch)
# -------------------------
in_ch = m
d_out = 2

global_avg_pool(x) = mean(x; dims=1)  # average over time => (1, channels, batch)

model = Chain(
    Conv((7,), in_ch => 32, relu; pad=3),
    Conv((7,), 32 => 64, relu; pad=3),
    Conv((5,), 64 => 64, relu; pad=2),
    Conv((3,), 64 => 64, relu; pad=1),
    global_avg_pool,                    # (1, 64, batch)
    x -> reshape(x, :, size(x, 3)),     # (64, batch)
    Dense(64, 64, relu),
    Dense(64, d_out),
    x -> softplus.(x)                   # ensure positive (lambda, mu)
)

# -------------------------
# Batching helpers
# -------------------------
function make_batch(Z_list::Vector{Matrix{Float64}}, theta_mat::AbstractMatrix, idxs)
    xb = cat((to_cnn_input(Z_list[i]) for i in idxs)...; dims=3)  # (T, m, B)
    yb = Float32.(theta_mat[:, idxs])                             # (2, B)
    return xb, yb
end

loss_fn(yhat, ytrue) = mean(abs.(yhat .- ytrue))  # MAE

function eval_mae(model, Z_list, theta_mat; batchsize::Int=64)
    K = size(theta_mat, 2)
    total = 0.0
    count = 0
    for i in 1:batchsize:K
        idxs = i:min(i+batchsize-1, K)
        xb, yb = make_batch(Z_list, theta_mat, idxs)
        yhat = model(xb)
        total += mean(abs.(Float64.(yhat) .- Float64.(yb)))
        count += 1
    end
    return total / count
end

# -------------------------
# Train (NEW Flux optimizer API)
# -------------------------
opt_state = Flux.setup(Adam(1e-3), model)

K_train = size(theta_train64, 2)
epochs = 30
batchsize = 64   # T=1281 is large; reduce if RAM is tight (try 32)

println("\nTraining CNN1D...")
for epoch in 1:epochs
    idx = randperm(K_train)

    for i in 1:batchsize:K_train
        idxs = idx[i:min(i+batchsize-1, K_train)]
        xb, yb = make_batch(Z_train64, theta_train64, idxs)

        _, grads = Flux.withgradient(model) do m
            yhat = m(xb)
            loss_fn(yhat, yb)
        end

        Flux.update!(opt_state, model, grads[1])
    end

    train_mae = eval_mae(model, Z_train64, theta_train64; batchsize=batchsize)
    val_mae   = eval_mae(model, Z_val64,   theta_val64;   batchsize=batchsize)
    println("Epoch $epoch | train MAE = $(round(train_mae, digits=6)) | val MAE = $(round(val_mae, digits=6))")
end

println("Training finished.")

# -------------------------
# Test: MAE for lambda and mu separately
# -------------------------
K_test = size(theta_test64, 2)
pred = Matrix{Float64}(undef, 2, K_test)

for i in 1:batchsize:K_test
    idxs = i:min(i+batchsize-1, K_test)
    xb, _ = make_batch(Z_test64, theta_test64, idxs)
    yhat = Float64.(model(xb))  # (2, B)
    pred[:, idxs] .= yhat
end

mae_lambda = mean(abs.(pred[1, :] .- theta_test64[1, :]))
mae_mu     = mean(abs.(pred[2, :] .- theta_test64[2, :]))

println("\n=== TEST MAE (CNN1D) ===")
println("MAE(lambda) = ", mae_lambda)
println("MAE(mu)     = ", mae_mu)