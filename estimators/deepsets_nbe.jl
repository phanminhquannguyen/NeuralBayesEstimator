# ============================================================
# deepsets_nbe.jl
#
# DeepSets neural Bayes estimator (NBE): a single (psi, phi) DeepSets
# network trained to predict the two rate parameters (e.g. (lambda, mu))
# directly and jointly from a trajectory -- unlike sr_deepsets.jl, which
# splits the target into two separately-trained networks (r = lambda-mu,
# log_s = log(lambda+mu)), this is one network, one joint MAE loss.
#
# Usage:
#   julia --project=. estimators/deepsets_nbe.jl <extinct|non_extinct|non_extinct_Q|non_extinct_SIS|extinct_SIS>
#
# The core functions (build_deepsets_nbe, fit_deepsets_nbe,
# train_and_evaluate_deepsets_nbe) are reusable: `include` this file from
# another script to train/evaluate on in-memory data without going through
# the data/*.jld2 + CLI round trip (see experiments/chain_length_sweep.jl).
# ============================================================

using Random
using JLD2
using Flux
using NeuralEstimators
using Statistics

# ------------------------------------------------------------
# Preprocessing: log1p, clip to robust quantile range, median/MAD scale
# (same recipe as sr_deepsets.jl's R-network preprocessing)
# ------------------------------------------------------------
function fit_global_robust_normalizer(Z_train::Vector{Matrix{Float64}};
                                      qlo::Float64=0.001, qhi::Float64=0.999, eps::Float64=1e-6)
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

    return (lo=lo, hi=hi, med=med, mad=mad, eps=eps)
end

function preprocess_log_clip_robust(Zk::AbstractMatrix{<:Real}, norm)
    X = log1p.(Float64.(Zk))
    X = clamp.(X, norm.lo, norm.hi)
    return (X .- norm.med) ./ norm.mad
end

preprocess_dataset(Z_list, norm) = [preprocess_log_clip_robust(Zk, norm) for Zk in Z_list]

# ------------------------------------------------------------
# Model: DeepSet(psi, phi) -> 2 positive outputs (softplus)
# ------------------------------------------------------------
function build_deepsets_nbe(n_in::Int; w::Int=64, d_out::Int=2)
    psi = Chain(Dense(n_in, w, relu), Dense(w, w, relu), Dense(w, w, relu))
    phi = Chain(Dense(w, w, relu), Dense(w, w, relu), Dense(w, d_out), x -> softplus.(x))
    return PointEstimator(DeepSet(psi, phi))
end

# ------------------------------------------------------------
# L2-regularised training (same pattern as sr_deepsets.jl's train_with_l2):
# collect trainables ONCE, never Flux.params inside the loss closure.
# ------------------------------------------------------------
l2_penalty(trainables; weight::Float64=1e-6) =
    weight * sum(p -> sum(abs2, p), trainables)

function fit_deepsets_nbe(estimator, theta_train, theta_val, Z_train, Z_val;
                          epochs::Int=40, batchsize::Int=1024, lr::Float64=1e-3,
                          weight_decay::Float64=1e-6, use_gpu::Bool=false, verbose::Bool=true)
    trainables = Flux.trainables(estimator)
    opt = Flux.setup(Adam(lr), estimator)
    loss_fn = (yhat, ytrue) -> Flux.mae(yhat, ytrue) + l2_penalty(trainables; weight=weight_decay)

    return train(
        estimator, theta_train, theta_val, Z_train, Z_val;
        epochs=epochs, batchsize=batchsize, loss=loss_fn, optimiser=opt,
        use_gpu=use_gpu, verbose=verbose
    )
end

mae_by_row(pred::AbstractMatrix, truth::AbstractMatrix) = vec(mean(abs.(pred .- truth); dims=2))

# ------------------------------------------------------------
# End-to-end: fit the normalizer + model on train/val, evaluate on test.
# theta_* are 2xK, Z_*_raw are Vector{Matrix} (T x m), untransformed.
# ------------------------------------------------------------
function train_and_evaluate_deepsets_nbe(theta_train, theta_val, theta_test,
                                         Z_train_raw, Z_val_raw, Z_test_raw;
                                         epochs::Int=40, batchsize::Int=1024, w::Int=64,
                                         verbose::Bool=true)
    norm = fit_global_robust_normalizer(Z_train_raw)
    Z_train = preprocess_dataset(Z_train_raw, norm)
    Z_val   = preprocess_dataset(Z_val_raw,   norm)
    Z_test  = preprocess_dataset(Z_test_raw,  norm)

    n_in = size(Z_train[1], 1)
    estimator = build_deepsets_nbe(n_in; w=w)

    estimator = fit_deepsets_nbe(
        estimator, Float64.(theta_train), Float64.(theta_val), Z_train, Z_val;
        epochs=epochs, batchsize=batchsize, verbose=verbose
    )

    pred_test = Float64.(NeuralEstimators.estimate(estimator, Z_test; use_gpu=false))
    mae = mae_by_row(pred_test, Float64.(theta_test))

    return (estimator=estimator, norm=norm, pred_test=pred_test, mae=mae)
end

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1 || !(ARGS[1] in ["extinct", "non_extinct", "non_extinct_Q", "non_extinct_SIS", "extinct_SIS"])
        println("Usage: julia --project=. estimators/deepsets_nbe.jl <extinct|non_extinct|non_extinct_Q|non_extinct_SIS|extinct_SIS>")
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

    @load data_file theta_train theta_val theta_test Z_train Z_val Z_test times m T

    theta_train64 = Float64.(theta_train)
    theta_val64   = Float64.(theta_val)
    theta_test64  = Float64.(theta_test)
    Z_train_raw = [Float64.(Zk) for Zk in Z_train]
    Z_val_raw   = [Float64.(Zk) for Zk in Z_val]
    Z_test_raw  = [Float64.(Zk) for Zk in Z_test]

    println("Loaded: ", data_file)
    println("T=", T, " | m=", m,
            " | K_train=", size(theta_train64, 2),
            " | K_val=", size(theta_val64, 2),
            " | K_test=", size(theta_test64, 2))

    result = train_and_evaluate_deepsets_nbe(
        theta_train64, theta_val64, theta_test64, Z_train_raw, Z_val_raw, Z_test_raw;
        epochs=40, batchsize=1024, verbose=true
    )

    println("\n========== TEST PERFORMANCE (DeepSets NBE) ==========")
    println("MAE(param 1) = ", round(result.mae[1], digits=6))
    println("MAE(param 2) = ", round(result.mae[2], digits=6))
    println("=======================================================\n")
end
