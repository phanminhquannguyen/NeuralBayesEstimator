# ============================================================
# test_data_simulator.jl
# Generate fresh TEST data (many different theta) and save it.
#
# Run:
#   julia test_data_simulator.jl
#
# Optional:
#   julia test_data_simulator.jl <output_test_data.jld2>
# ============================================================

using Random
using Dates
using Printf
using JLD2
using Distributions

# This file should define:
#   simulate_bdp(times, z0, mu, lambda)
#   simulate_bdp_sub_nonextinct(times, z0, mu, lambda; ...)
include("simulator.jl")

# ------------------------------------------------------------
# Prior sampler (Θ is 2×K with rows (lambda, mu))
# ------------------------------------------------------------
function prior_sampler(K::Int; r_max::Real = 5.0)
    lambda = rand(Uniform(0, r_max), K)
    mu     = rand(Uniform(0, r_max), K)
    return vcat(reshape(lambda, 1, :), reshape(mu, 1, :))  # 2×K
end

# ------------------------------------------------------------
# Simulate fresh datasets for NBE:
# returns Vector{Matrix{Int}} of length K
# each Z_list[k] is T×m (columns = trajectories/replicates)
# ------------------------------------------------------------
function simulate_test_sets(theta::AbstractMatrix, m::Int, times;
                            non_extinct::Bool = false,
                            z0_range = 1:10,
                            branch::Int = 2,
                            survival_target::Real = 0.5,
                            max_restarts::Int = 1000)

    K = size(theta, 2)
    T = length(times)
    Z_list = Vector{Matrix{Int}}(undef, K)

    for k in 1:K
        lambda = theta[1, k]
        mu     = theta[2, k]
        z0 = rand(z0_range)

        Zk = Matrix{Int}(undef, T, m)

        if non_extinct && lambda < mu
            # subcritical: condition on non-extinction using splitting
            for i in 1:m
                Zk[:, i] = simulate_bdp_sub_nonextinct(
                    times, z0, mu, lambda;
                    branch = branch,
                    survival_target = survival_target,
                    max_restarts = max_restarts
                ).Z
            end
        else
            # unconditional: extinction allowed
            for i in 1:m
                Zk[:, i] = simulate_bdp(times, z0, mu, lambda).Z
            end
        end

        Z_list[k] = Zk
    end

    return Z_list
end

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
function main()
    out_test_file = length(ARGS) >= 1 ? ARGS[1] : "test_data.jld2"

    Random.seed!(123)

    # Use the SAME grid as training (adjust if needed)
    times = collect(0.0:0.1:1.9)
    T = length(times)

    # test sizes
    K_test = 2000
    m      = 20
    r_max  = 5.0

    # whether to condition subcritical trajectories on survival
    non_extinct = false   # set true if you want subcritical to be non-extinct

    println("Generating fresh test data...")
    start_t = now()
    println("Start: ", start_t)

    theta_test = prior_sampler(K_test; r_max=r_max)
    Z_test     = simulate_test_sets(theta_test, m, times; non_extinct=non_extinct)

    end_t = now()
    println("End:   ", end_t)
    println("Elapsed: ", end_t - start_t)

    @printf("\nSaved objects:\n")
    @printf("  theta_test: 2×%d\n", K_test)
    @printf("  Z_test: %d datasets; each %d×%d (T×m)\n", length(Z_test), T, m)

    @save out_test_file theta_test Z_test times T m K_test r_max non_extinct

    println("\nSaved to: ", out_test_file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end