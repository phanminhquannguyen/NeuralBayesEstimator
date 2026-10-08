# ============================================================
# lbdp_discrete.jl
#
# Supercritical LBDP simulator (discrete grid) via Binomial + NegativeBinomial
# transitions -- an alternative to the continuous-time SSA in lbdp_ssa.jl.
#
# Usage:
#   julia --project=. simulators/lbdp_discrete.jl <mode> <M> <dt> <tmax> <K_train> <K_val> <K_test> <r_max>
#   (all args optional; mode = "extinct" | "non_extinct")
#
# Saves: data/lbdp_discrete_extinct.jld2 | data/lbdp_discrete_non_extinct.jld2
# ============================================================

using Random
using Distributions
using JLD2

include("common.jl")

# Same delta concept as lbdp_ssa.jl
const DELTA = 2.0

sample_theta(K::Int; r_max::Real = 6.0, rng::AbstractRNG = Random.default_rng()) =
    sample_theta_constrained(K; r_max = r_max, delta = DELTA, cmp = >, rng = rng)

# ------------------------------------------------------------
# alpha(dt), beta(dt) for LBDP (one ancestor)
# ------------------------------------------------------------
function alpha_beta(time_step::Float64, birth_rate::Float64, death_rate::Float64)
    @assert time_step ≥ 0
    @assert birth_rate ≥ 0 && death_rate ≥ 0

    # Pure birth
    if death_rate == 0.0
        alpha = 0.0
        beta  = 1.0 - exp(-birth_rate * time_step)
        return (alpha, beta)
    end

    # Critical: birth_rate == death_rate (limit)
    if abs(birth_rate - death_rate) < 1e-12
        alpha = (birth_rate * time_step) / (1.0 + birth_rate * time_step)
        beta  = alpha
        return (alpha, beta)
    end

    omega = birth_rate - death_rate
    growth_factor = exp(omega * time_step)

    alpha = death_rate * (growth_factor - 1.0) / (birth_rate * growth_factor - death_rate)
    alpha = min(max(alpha, 0.0), 1.0)

    beta  = (birth_rate * alpha) / death_rate
    beta  = min(max(beta, 0.0), 1.0)

    return (alpha, beta)
end

# ------------------------------------------------------------
# One step transition: population -> next_population
# ------------------------------------------------------------
function step_transition(current_population::Int,
                         birth_rate::Float64,
                         death_rate::Float64,
                         time_step::Float64,
                         rng::AbstractRNG)

    if current_population == 0
        return 0
    end

    alpha, beta = alpha_beta(time_step, birth_rate, death_rate)

    survival_probability = 1.0 - alpha
    survival_probability = min(max(survival_probability, 0.0), 1.0)

    survivors = rand(rng, Binomial(current_population, survival_probability))
    if survivors == 0
        return 0
    end

    success_probability = 1.0 - beta
    success_probability = min(max(success_probability, 0.0), 1.0)

    extra = rand(rng, NegativeBinomial(survivors, success_probability))
    return survivors + extra
end

# ------------------------------------------------------------
# Simulate one trajectory on the grid (returns Vector{Int} length T)
# ------------------------------------------------------------
function simulate_one_path(times::Vector{Float64},
                           initial_population::Int,
                           birth_rate::Float64,
                           death_rate::Float64,
                           rng::AbstractRNG)

    T = length(times)
    path = Vector{Int}(undef, T)

    population = initial_population
    path[1] = population

    for t_index in 2:T
        time_step = times[t_index] - times[t_index - 1]
        population = step_transition(population, birth_rate, death_rate, time_step, rng)
        path[t_index] = population
    end

    return path
end

# ------------------------------------------------------------
# Simulate datasets for NBE (same output style as lbdp_ssa.jl)
# Theta: 2xK with rows (birth_rate, death_rate)
# Output: Z_list where each Zk is TxM matrix of Int
# ------------------------------------------------------------
function simulate_lbdp_for_NBE(Theta::AbstractMatrix,
                               m::Int,
                               times::Vector{Float64};
                               non_extinct::Bool=false,
                               initial_population::Int=5,
                               rng::AbstractRNG=Random.default_rng(),
                               max_restarts::Int=100000)

    K = size(Theta, 2)
    T = length(times)
    Z_list = Vector{Matrix{Int}}(undef, K)

    for k in 1:K
        birth_rate = Theta[1, k]
        death_rate = Theta[2, k]

        Zk = Matrix{Int}(undef, T, m)

        for rep in 1:m
            if non_extinct
                ok = false
                for _ in 1:max_restarts
                    path = simulate_one_path(times, initial_population, birth_rate, death_rate, rng)
                    if path[end] > 0
                        Zk[:, rep] = path
                        ok = true
                        break
                    end
                end
                ok || error("Failed to produce non-extinct path within max_restarts. Try smaller tmax or larger birth_rate - death_rate.")
            else
                Zk[:, rep] = simulate_one_path(times, initial_population, birth_rate, death_rate, rng)
            end
        end

        Z_list[k] = Zk
    end

    return Z_list
end

function main()
    # Defaults (same as original)
    mode    = "non_extinct"   # "extinct" or "non_extinct"
    M       = 1
    dt      = 0.1
    tmax    = 3
    K_train = 5000
    K_val   = 1000
    K_test  = 1000
    r_max   = 5.0
    seed    = 1

    # julia lbdp_discrete.jl <mode> <M> <dt> <tmax> <K_train> <K_val> <K_test> <r_max>
    if length(ARGS) >= 1; mode    = ARGS[1]; end
    if length(ARGS) >= 2; M       = parse(Int, ARGS[2]); end
    if length(ARGS) >= 3; dt      = parse(Float64, ARGS[3]); end
    if length(ARGS) >= 4; tmax    = parse(Float64, ARGS[4]); end
    if length(ARGS) >= 5; K_train = parse(Int, ARGS[5]); end
    if length(ARGS) >= 6; K_val   = parse(Int, ARGS[6]); end
    if length(ARGS) >= 7; K_test  = parse(Int, ARGS[7]); end
    if length(ARGS) >= 8; r_max   = parse(Float64, ARGS[8]); end

    non_extinct =
        lowercase(mode) in ["non_extinct", "nonextinct", "non-extinct", "survival"]

    Random.seed!(seed)
    rng = MersenneTwister(seed)

    times = collect(0.0:dt:tmax)
    T = length(times)
    m = M

    theta_train = sample_theta(K_train; r_max=r_max, rng=rng)
    theta_val   = sample_theta(K_val;   r_max=r_max, rng=rng)
    theta_test  = sample_theta(K_test;  r_max=r_max, rng=rng)

    println("Mode: ", mode, "  (non_extinct = ", non_extinct, ")")
    println("Grid: dt=", dt, ", tmax=", tmax, ", T=", T, " | m=", m)
    println("K_train=", K_train, " K_val=", K_val, " K_test=", K_test, " r_max=", r_max)
    println("Prior: birth_rate, death_rate ~ Uniform(1, ", r_max, "), with birth_rate > death_rate and abs(birth_rate - death_rate) ≤ ", DELTA)

    println("Simulating training data...")
    Z_train = simulate_lbdp_for_NBE(theta_train, m, times;
                                   non_extinct=non_extinct,
                                   initial_population=5,
                                   rng=rng)

    println("Simulating validation data...")
    Z_val   = simulate_lbdp_for_NBE(theta_val, m, times;
                                   non_extinct=non_extinct,
                                   initial_population=5,
                                   rng=rng)

    println("Simulating test data...")
    Z_test  = simulate_lbdp_for_NBE(theta_test, m, times;
                                   non_extinct=non_extinct,
                                   initial_population=5,
                                   rng=rng)

    mkpath("data")
    out_file = non_extinct ? "data/lbdp_discrete_non_extinct.jld2" : "data/lbdp_discrete_extinct.jld2"

    @save out_file theta_train theta_val theta_test Z_train Z_val Z_test times m T K_train K_val K_test r_max non_extinct

    println("Saved dataset to ", out_file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
