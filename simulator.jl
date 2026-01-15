
# Linear Birth–Death Process (LBDP) simulator by Gillespie SSA + optional non-extinction


# Load neccessary packages
using Random
using Dates
using Distributions
using JLD2


# Prior sampler: Theta is 2×K with rows (lambda, mu), we can change the r_max to adjust prior range
function sample_theta(K::Int; r_max::Real = 5.0)
    lambda = rand(Uniform(0, r_max), K)
    mu     = rand(Uniform(0, r_max), K)
    return vcat(reshape(lambda, 1, :), reshape(mu, 1, :))  # 2×K
end


#  Event-based Gillespie SSA on [0, horizon]
function simulate_interval_bdp(z0::Integer, lambda::Real, mu::Real, horizon::Real)
    z = Int(z0)
    t = 0.0

    times  = Float64[0.0]
    states = Int[z]

    while t < horizon && z > 0
        rate_total = (lambda + mu) * z
        rate_total <= 0 && break

        dt = randexp() / rate_total
        (t + dt > horizon) && break

        # choose event type
        if rand() < (lambda * z) / rate_total
            z += 1
        else
            z -= 1
        end

        t += dt
        push!(times, t)
        push!(states, z)
    end

    return (times = times, states = states)
end

# Sample event-based trajectory on a fixed grid
function sample_on_grid(grid_times::AbstractVector{<:Real},
                        event_times::AbstractVector{<:Real},
                        event_states::AbstractVector{<:Integer})
    Z = Vector{Int}(undef, length(grid_times))
    @inbounds for i in eachindex(grid_times)
        j = searchsortedlast(event_times, grid_times[i])
        j = max(j, 1)
        Z[i] = event_states[j]
    end
    return Z
end


# Unconditional simulation (extinction allowed)
function simulate_bdp(times::AbstractVector{<:Real},
                      z0::Integer, mu::Real, lambda::Real)
    tmax = maximum(times)
    seg = simulate_interval_bdp(z0, lambda, mu, tmax)

    if last(seg.states) == 0 && last(seg.times) < tmax
        push!(seg.times, float(tmax))
        push!(seg.states, 0)
    end

    Z = sample_on_grid(times, seg.times, seg.states)
    return (time = collect(times), Z = Z)
end


# Subcritical NON-EXTINCT simulation via multilevel splitting

function simulate_bdp_sub_nonextinct(times::AbstractVector{<:Real},
                                     z0::Integer, mu::Real, lambda::Real;
                                     branch::Integer = 2,
                                     survival_target::Real = 0.5,
                                     max_restarts::Integer = 1000)

    rho = lambda - mu   # < 0 for subcritical
    tmax = maximum(times)

    s_opt = log(survival_target) / rho
    n = max(1, Int(round(tmax / s_opt)))
    s = tmax / n

    for _restart in 1:max_restarts
        # Level 1: simulate until s, conditioned on survival
        seg1 = simulate_interval_bdp(z0, lambda, mu, s)
        while last(seg1.states) == 0
            seg1 = simulate_interval_bdp(z0, lambda, mu, s)
        end

        if n == 1
            Z = sample_on_grid(times, seg1.times, seg1.states)
            return (time = collect(times), Z = Z)
        end

        # Storage:
        mat_states = Vector{Any}(undef, n)
        mat_times  = Vector{Any}(undef, n)
        mat_parent = Vector{Any}(undef, n)

        mat_states[1] = seg1.states
        mat_times[1]  = seg1.times
        mat_parent[1] = 0

        # v_end is a VECTOR of end populations for current level
        v_end = [last(seg1.states)]

        it = 1
        while it < n && !isempty(v_end)
            # end pops for all paths at level it
            pops = (it == 1) ? v_end : [last(st) for st in mat_states[it]]

            next_states = Vector{Vector{Int}}()
            next_times  = Vector{Vector{Float64}}()
            next_parent = Int[]
            next_end    = Int[]

            for (d, zstart) in enumerate(pops)
                for _ in 1:branch
                    seg = simulate_interval_bdp(zstart, lambda, mu, s)
                    if last(seg.states) > 0
                        push!(next_states, seg.states)
                        push!(next_times, (it*s) .+ seg.times)
                        push!(next_parent, d)
                        push!(next_end, last(seg.states))
                    end
                end
            end

            # If no survivors, force failure and restart
            if isempty(next_end)
                v_end = Int[]
                break
            end

            # Advance to next level ONLY after we know it exists
            it += 1
            mat_states[it] = next_states
            mat_times[it]  = next_times
            mat_parent[it] = next_parent
            v_end = next_end
        end

        # success: reached level n with at least one survivor path
        if it == n && !isempty(v_end)
            k = 1  # pick the first survivor
            full_states = mat_states[n][k]
            full_times  = mat_times[n][k]
            mother = mat_parent[n][k]

            if n >= 3
                for lev in (n-1):-1:2
                    full_states = vcat(mat_states[lev][mother], full_states)
                    full_times  = vcat(mat_times[lev][mother],  full_times)
                    mother = mat_parent[lev][mother]
                end
            end

            # prepend level 1
            full_states = vcat(mat_states[1], full_states)
            full_times  = vcat(mat_times[1],  full_times)

            Z = sample_on_grid(times, full_times, full_states)
            return (time = collect(times), Z = Z)
        end
    end

    error("Failed to produce a non-extinct subcritical trajectory within max_restarts.")
end


# Simulate datasets for NBE
function simulate_lbdp_for_NBE(Theta::AbstractMatrix, m::Int, times;
                               non_extinct::Bool = false,
                               z0_range = 1:10,
                               branch::Int = 2,
                               survival_target::Real = 0.5,
                               max_restarts::Int = 1000)

    K = size(Theta, 2)
    T = length(times)
    Z_list = Vector{Matrix{Int}}(undef, K)

    for k in 1:K
        lambda = Theta[1, k]
        mu     = Theta[2, k]

        # random initial population (shared across the m replicates for this theta)
        z0 = rand(z0_range)

        Zk = Matrix{Int}(undef, T, m)

        # Choose simulator:
        # - if non_extinct=true AND subcritical (lambda < mu), enforce survival via splitting
        # - otherwise simulate normally (extinction allowed)
        if non_extinct && lambda < mu
            for i in 1:m
                Zk[:, i] = simulate_bdp_sub_nonextinct(
                    times, z0, mu, lambda;
                    branch = branch,
                    survival_target = survival_target,
                    max_restarts = max_restarts
                ).Z
            end
        else
            for i in 1:m
                Zk[:, i] = simulate_bdp(times, z0, mu, lambda).Z
            end
        end

        Z_list[k] = Zk
    end

    return Z_list
end


# generate train / val / test datasets and save
function main()
    # -----------------------------
    # Defaults (same as your current)
    # -----------------------------
    mode    = "non_extinct"   # "extinct" or "non_extinct"
    M       = 20
    dt      = 0.1
    tmax    = 1.9
    K_train = 10_000
    K_val   = 2_000
    K_test  = 2_000
    r_max   = 5.0
    seed    = 1


    # Usage: (must provide in order and all aruguments)
    # julia simulator.jl <mode> <M> <dt> <tmax> <K_train> <K_val> <K_test> <r_max>

    if length(ARGS) >= 1
        mode = ARGS[1]
    end
    if length(ARGS) >= 2
        M = parse(Int, ARGS[2])
    end
    if length(ARGS) >= 3
        dt = parse(Float64, ARGS[3])
    end
    if length(ARGS) >= 4
        tmax = parse(Float64, ARGS[4])
    end
    if length(ARGS) >= 5
        K_train = parse(Int, ARGS[5])
    end
    if length(ARGS) >= 6
        K_val = parse(Int, ARGS[6])
    end
    if length(ARGS) >= 7
        K_test = parse(Int, ARGS[7])
    end
    if length(ARGS) >= 8
        r_max = parse(Float64, ARGS[8])
    end

    # mode -> non_extinct flag
    non_extinct =
        lowercase(mode) in ["non_extinct", "nonextinct", "non-extinct", "survival"]

    Random.seed!(seed)

    times = collect(0.0:dt:tmax)
    T = length(times)

    m = M

    theta_train = sample_theta(K_train; r_max = r_max)
    theta_val   = sample_theta(K_val;   r_max = r_max)
    theta_test  = sample_theta(K_test;  r_max = r_max)

    println("Mode: ", mode, "  (non_extinct = ", non_extinct, ")")
    println("Grid: dt=", dt, ", tmax=", tmax, ", T=", T, " | m=", m)
    println("K_train=", K_train, " K_val=", K_val, " K_test=", K_test, " r_max=", r_max)

    println("Simulating training data...")
    Z_train = simulate_lbdp_for_NBE(theta_train, m, times; non_extinct = non_extinct)

    println("Simulating validation data...")
    Z_val   = simulate_lbdp_for_NBE(theta_val, m, times; non_extinct = non_extinct)

    println("Simulating test data...")
    Z_test  = simulate_lbdp_for_NBE(theta_test, m, times; non_extinct = non_extinct)

    # output name depends on mode
    out_file = non_extinct ? "nbe_lbdp_non_extinct.jld2" : "nbe_lbdp_extinct.jld2"

    @save out_file theta_train theta_val theta_test Z_train Z_val Z_test times m T K_train K_val K_test r_max non_extinct

    println("Saved dataset to ", out_file)
end