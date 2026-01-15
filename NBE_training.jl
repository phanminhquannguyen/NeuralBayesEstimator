# ============================================================
# nbe_training.jl
# Train DeepSets Neural Bayes Estimator on LBDP data
#
# BASIC USAGE (recommended for normal runs):
#   julia nbe_training.jl extinct
#   julia nbe_training.jl non_extinct
#
# This will:
#   - Load the default dataset:
#       extinct     → nbe_lbdp_extinct.jld2
#       non_extinct → nbe_lbdp_non_extinct.jld2
#   - Train the DeepSets Neural Bayes Estimator
#   - Save the trained model to:
#       trained_estimator_extinct.jld2
#       trained_estimator_non_extinct.jld2
#
# ADVANCED USAGE (for pipelines / sweeps):
#   julia nbe_training.jl <extinct|non_extinct> <data_file.jld2> <model_file.jld2>
#
# Example:
#   julia nbe_training.jl extinct sweeps/run1/data_extinct.jld2 sweeps/run1/trained_extinct.jld2
#
# ============================================================

using Random
using JLD2
using Flux
using NeuralEstimators

# ------------------------------------------------------------
# 1) Choose case from CLI (optional: data_file, model_file)
# ------------------------------------------------------------
if length(ARGS) < 1 || !(ARGS[1] in ["extinct", "non_extinct"])
    println("Usage: julia nbe_training.jl <extinct|non_extinct> [data_file.jld2] [model_file.jld2]")
    exit(1)
end

case = ARGS[1]

# Defaults (if not provided)
default_data_file  = case == "extinct" ? "nbe_lbdp_extinct.jld2" : "nbe_lbdp_non_extinct.jld2"
default_model_file = case == "extinct" ? "trained_estimator_extinct.jld2" : "trained_estimator_non_extinct.jld2"

data_file  = length(ARGS) >= 2 ? ARGS[2] : default_data_file
model_file = length(ARGS) >= 3 ? ARGS[3] : default_model_file

# ------------------------------------------------------------
# 2) Load data
# ------------------------------------------------------------
@load data_file theta_train theta_val theta_test Z_train Z_val Z_test times m T

# Type consistency
theta_train64 = Float64.(theta_train)
theta_val64   = Float64.(theta_val)

Z_train64 = [Float64.(Zk) for Zk in Z_train]
Z_val64   = [Float64.(Zk) for Zk in Z_val]

# ------------------------------------------------------------
# 3) Build DeepSets architecture
# ------------------------------------------------------------
d = 2     # dimension of theta (lambda, mu)
n = T     # length of each trajectory on the grid
w = 64

psi = Chain(
    Dense(n, w, relu),
    Dense(w, w, relu),
    Dense(w, w, relu)
)

phi = Chain(
    Dense(w, w, relu),
    Dense(w, w, relu),
    Dense(w, d, softplus)   # keep outputs positive
)

network   = DeepSet(psi, phi)
estimator = PointEstimator(network)

# ------------------------------------------------------------
# 4) Train
# ------------------------------------------------------------
epochs    = 40
batchsize = 32

opt = Flux.setup(Adam(1e-3), estimator)

estimator = train(
    estimator,
    theta_train64,
    theta_val64,
    Z_train64,
    Z_val64;
    epochs    = epochs,
    batchsize = batchsize,
    loss      = Flux.mae,
    optimiser = opt,
    use_gpu   = false,
    verbose   = true
)

println("\nTraining finished.")

# ------------------------------------------------------------
# 5) Save trained estimator
# ------------------------------------------------------------
@save model_file estimator times T m case
println("Saved trained model to: ", model_file)