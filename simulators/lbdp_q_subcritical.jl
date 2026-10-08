# ============================================================
# lbdp_q_subcritical.jl  (SUBCRITICAL ONLY)
#
# Simulate non-extinct datasets for NBE using:
#   - Subcritical (lambda < mu): Q-process simulation (non-extinct by construction)
#
# Saves: data/lbdp_q_subcritical_non_extinct.jld2
#
# Run (defaults):
#   julia --project=. simulators/lbdp_q_subcritical.jl
#
# Optional args:
#   julia --project=. simulators/lbdp_q_subcritical.jl <M> <dt> <tmax> <K_train> <K_val> <K_test> <r_max> <delta> <seed>
# ============================================================

using Random
using Distributions
using JLD2

include("common.jl")

# Prior sampler: Theta is 2xK with rows (lambda, mu). Subcritical only:
# lambda < mu, and |lambda - mu| <= delta (kept for consistency with the
# other simulators).
sample_theta_subcritical(K::Int; r_max::Real = 6.0, delta::Real = 2.0) =
    sample_theta_constrained(K; r_max = r_max, delta = delta, cmp = <)

# ------------------------------------------------------------
# Q-process simulator for subcritical binary LBDP (lambda < mu)
# Generator:
#   i -> i+1 at rate (i+1)*lambda
#   i -> i-1 at rate (i-1)*mu   (death disabled at i=1)
# ------------------------------------------------------------
function simulate_interval_bdp_Q(z0::Integer, λ::Real, μ::Real, horizon::Real;
                                 max_events::Int = 2_000_000,
                                 max_state::Int = 50_000)
    z = Int(z0)
    t = 0.0

    times  = Float64[0.0]
    states = Int[z]
    nevents = 0

    while t < horizon
        nevents += 1
        if nevents > max_events || z > max_state
            break
        end

        rate_birth = (z + 1) * λ
        rate_death = (z > 1) ? (z - 1) * μ : 0.0
        rate_total = rate_birth + rate_death
        rate_total <= 0 && break

        dt = randexp() / rate_total
        (t + dt > horizon) && break

        if rand() < rate_birth / rate_total
            z += 1
        else
            z -= 1   # cannot hit 0 since rate_death = 0 at z=1
        end

        t += dt
        push!(times, t)
        push!(states, z)
    end

    return (times = times, states = states)
end

# ------------------------------------------------------------
# Discretely observed Q-process trajectory on grid
# ------------------------------------------------------------
function simulate_bdp_Q(times::AbstractVector{<:Real}, z0::Integer, λ::Real, μ::Real)
    tmax = maximum(times)
    seg  = simulate_interval_bdp_Q(z0, λ, μ, tmax)
    Z    = sample_on_grid(times, seg.times, seg.states)
    return (time = collect(times), Z = Z)
end

# ------------------------------------------------------------
# Simulate datasets for NBE (SUBCRITICAL ONLY, Q-process only)
# ------------------------------------------------------------
function simulate_lbdp_for_NBE_Q_subcritical(Theta::AbstractMatrix, m::Int, times;
                                             z0::Int = 5)
    K = size(Theta, 2)
    T = length(times)
    Z_list = Vector{Matrix{Int}}(undef, K)

    for k in 1:K
        λ = Theta[1, k]
        μ = Theta[2, k]

        # Safety check: ensure sampler did its job
        (λ < μ) || error("Found non-subcritical params at k=$k: λ=$λ, μ=$μ")

        Zk = Matrix{Int}(undef, T, m)
        for i in 1:m
            Zk[:, i] = simulate_bdp_Q(times, z0, λ, μ).Z
        end
        Z_list[k] = Zk
    end

    return Z_list
end

# ------------------------------------------------------------
# Main: generate train/val/test and save
# ------------------------------------------------------------
function main()
    # Defaults
    M       = 1
    dt      = 0.1
    tmax    = 256
    K_train = 5000
    K_val   = 1000
    K_test  = 1000
    r_max   = 5.0
    delta   = 2.0
    seed    = 1

    # Optional args:
    # julia lbdp_q_subcritical.jl <M> <dt> <tmax> <K_train> <K_val> <K_test> <r_max> <delta> <seed>
    if length(ARGS) >= 1; M       = parse(Int, ARGS[1]); end
    if length(ARGS) >= 2; dt      = parse(Float64, ARGS[2]); end
    if length(ARGS) >= 3; tmax    = parse(Float64, ARGS[3]); end
    if length(ARGS) >= 4; K_train = parse(Int, ARGS[4]); end
    if length(ARGS) >= 5; K_val   = parse(Int, ARGS[5]); end
    if length(ARGS) >= 6; K_test  = parse(Int, ARGS[6]); end
    if length(ARGS) >= 7; r_max   = parse(Float64, ARGS[7]); end
    if length(ARGS) >= 8; delta   = parse(Float64, ARGS[8]); end
    if length(ARGS) >= 9; seed    = parse(Int, ARGS[9]); end

    Random.seed!(seed)

    times = collect(0.0:dt:tmax)
    T = length(times)
    m = M
    z0 = 5

    theta_train = sample_theta_subcritical(K_train; r_max = r_max, delta = delta)
    theta_val   = sample_theta_subcritical(K_val;   r_max = r_max, delta = delta)
    theta_test  = sample_theta_subcritical(K_test;  r_max = r_max, delta = delta)

    println("============================================================")
    println("NON-EXTINCT DATA (SUBCRITICAL ONLY via Q-process)")
    println("Grid: dt=", dt, ", tmax=", tmax, ", T=", T, " | m=", m, " | z0=", z0)
    println("K_train=", K_train, " K_val=", K_val, " K_test=", K_test,
            " r_max=", r_max, " delta=", delta, " seed=", seed)
    println("============================================================")

    println("Simulating training data (Q-process)...")
    Z_train = simulate_lbdp_for_NBE_Q_subcritical(theta_train, m, times; z0 = z0)

    println("Simulating validation data (Q-process)...")
    Z_val   = simulate_lbdp_for_NBE_Q_subcritical(theta_val, m, times; z0 = z0)

    println("Simulating test data (Q-process)...")
    Z_test  = simulate_lbdp_for_NBE_Q_subcritical(theta_test, m, times; z0 = z0)

    mkpath("data")
    out_file = "data/lbdp_q_subcritical_non_extinct.jld2"

    non_extinct = true
    subcritical_only = true
    @save out_file theta_train theta_val theta_test Z_train Z_val Z_test times m T K_train K_val K_test r_max non_extinct subcritical_only delta seed z0

    println("Saved dataset to ", out_file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
