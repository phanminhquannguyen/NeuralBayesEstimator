# ============================================================
# chain_length_sweep.jl
#
# Does a longer observed trajectory help the DeepSets NBE
# (estimators/deepsets_nbe.jl) estimate (lambda, mu) more accurately?
# Trains/evaluates it at a range of trajectory lengths ("observations"),
# for the subcritical (Q-process) regime first, then the supercritical
# (SSA) regime.
#
# Subcritical: dt is fixed, tmax grows. The Q-process is conditioned to
#   survive, so its population stays bounded in distribution -- a longer
#   time horizon just means more informative observations.
# Supercritical: tmax is fixed, dt shrinks instead. Population here grows
#   like exp((lambda-mu)*t), so stretching tmax out the same way would
#   make the simulation blow up; sampling the same bounded horizon more
#   finely is the correct analogue of "a longer chain" here.
#
# Usage:
#   julia --project=. experiments/chain_length_sweep.jl
#
# Saves: chain_length_sweep_results.csv
# ============================================================

using Random
using Statistics
using DataFrames
using CSV

include("../simulators/lbdp_q_subcritical.jl")
include("../simulators/lbdp_ssa.jl")
include("../estimators/deepsets_nbe.jl")

const R_MAX      = 5.0
const DELTA      = 2.0
const Z0         = 5
const K_TRAIN    = 3000
const K_VAL      = 500
const K_TEST     = 500
const EPOCHS     = 25
const BATCHSIZE  = 256

function run_one(regime::Symbol, times::Vector{Float64}, seed::Int)
    Random.seed!(seed)

    if regime == :subcritical
        theta_train = sample_theta_subcritical(K_TRAIN; r_max = R_MAX, delta = DELTA)
        theta_val   = sample_theta_subcritical(K_VAL;   r_max = R_MAX, delta = DELTA)
        theta_test  = sample_theta_subcritical(K_TEST;  r_max = R_MAX, delta = DELTA)

        Z_train = simulate_lbdp_for_NBE_Q_subcritical(theta_train, 1, times; z0 = Z0)
        Z_val   = simulate_lbdp_for_NBE_Q_subcritical(theta_val,   1, times; z0 = Z0)
        Z_test  = simulate_lbdp_for_NBE_Q_subcritical(theta_test,  1, times; z0 = Z0)
    elseif regime == :supercritical
        theta_train = sample_theta(K_TRAIN; r_max = R_MAX, delta = DELTA)
        theta_val   = sample_theta(K_VAL;   r_max = R_MAX, delta = DELTA)
        theta_test  = sample_theta(K_TEST;  r_max = R_MAX, delta = DELTA)

        Z_train = simulate_lbdp_for_NBE(theta_train, 1, times; non_extinct = true)
        Z_val   = simulate_lbdp_for_NBE(theta_val,   1, times; non_extinct = true)
        Z_test  = simulate_lbdp_for_NBE(theta_test,  1, times; non_extinct = true)
    else
        error("unknown regime: $regime")
    end

    Z_train_raw = [Float64.(Zk) for Zk in Z_train]
    Z_val_raw   = [Float64.(Zk) for Zk in Z_val]
    Z_test_raw  = [Float64.(Zk) for Zk in Z_test]

    t0 = time()
    result = train_and_evaluate_deepsets_nbe(
        theta_train, theta_val, theta_test, Z_train_raw, Z_val_raw, Z_test_raw;
        epochs = EPOCHS, batchsize = BATCHSIZE, verbose = false
    )
    elapsed = time() - t0

    return (mae_lambda = result.mae[1], mae_mu = result.mae[2], elapsed = elapsed)
end

function sweep(regime::Symbol, observations_list::Vector{Int}, seed_base::Int;
               dt_fixed::Union{Nothing,Float64} = nothing,
               tmax_fixed::Union{Nothing,Float64} = nothing)
    rows = DataFrame(regime = String[], observations = Int[], T = Int[],
                     dt = Float64[], tmax = Float64[],
                     mae_lambda = Float64[], mae_mu = Float64[], seconds = Float64[])

    for (i, obs) in enumerate(observations_list)
        dt, tmax =
            dt_fixed !== nothing   ? (dt_fixed, obs * dt_fixed) :
            tmax_fixed !== nothing ? (tmax_fixed / obs, tmax_fixed) :
                                     error("must fix either dt or tmax")

        times = collect(0.0:dt:tmax)
        T = length(times)

        println("\n-- $(regime): observations=$obs (T=$T, dt=$(round(dt, digits=6)), tmax=$(round(tmax, digits=3))) --")
        r = run_one(regime, times, seed_base + i)
        println("   MAE(lambda) = ", round(r.mae_lambda, digits = 4),
                " | MAE(mu) = ", round(r.mae_mu, digits = 4),
                " | ", round(r.elapsed, digits = 1), "s")

        push!(rows, (String(regime), obs, T, dt, tmax, r.mae_lambda, r.mae_mu, r.elapsed))
    end

    return rows
end

observations_list = [40, 80, 160, 320, 640]

println("="^70)
println("SUBCRITICAL (Q-process): dt=0.1 fixed, tmax grows")
println("="^70)
rows_sub = sweep(:subcritical, observations_list, 1000; dt_fixed = 0.1)

println("\n" * "="^70)
println("SUPERCRITICAL (SSA): tmax=6 fixed, dt shrinks")
println("="^70)
rows_super = sweep(:supercritical, observations_list, 2000; tmax_fixed = 6.0)

rows = vcat(rows_sub, rows_super)
CSV.write("chain_length_sweep_results.csv", rows)

println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println(rows)
println("\nSaved: chain_length_sweep_results.csv")
