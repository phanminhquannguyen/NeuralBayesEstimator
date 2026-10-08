# ============================================================
# sis.jl
#
# Generate SIS / Verhulst PSDBDP dataset in DeepSets format with CLI mode:
#   julia --project=. simulators/sis.jl non_extinct
#   julia --project=. simulators/sis.jl extinct
#
# - non_extinct: if a simulated path goes extinct (hits 0 before tmax), resimulate
# - extinct:     if a simulated path does NOT go extinct by tmax, resimulate
#
# Saved file contains:
#   theta_train, theta_val, theta_test :: (2 x K)  [gamma; nu]
#   Z_train,     Z_val,     Z_test     :: Vector{Matrix{Float64}} each (T x m)
#   times, T, m, dt, tmax, N, z0, priors, seed, mode
# ============================================================

using Random
using JLD2

# CLI argument parsing
if length(ARGS) < 1 || !(ARGS[1] in ["non_extinct", "extinct"])
    println("Usage: julia simulator_SIS.jl <non_extinct|extinct>")
    exit(1)
end
mode = ARGS[1]

# Default settings for the simulator
m       = 1
dt      = 0.1
tmax    = 16
K_train = 5000
K_val   = 1000
K_test  = 1000
seed    = 1

# Model constants
N  = 100
z0 = 10

# Priors
gamma_min, gamma_max = 0.3, 7.5
nu_min,    nu_max    = 0.1, 5.5

# Optional: avoid too many fast extinctions by forcing gamma > nu
enforce_gamma_gt_nu = false

# Output file
out_file = mode == "non_extinct" ? "data/sis_non_extinct.jld2" : "data/sis_extinct.jld2"
# Create folder data if not exist yet
mkpath("data")

# Time grid
times = collect(0.0:dt:tmax)
T = length(times)
rng = MersenneTwister(seed)

# ------------------------------
# SIS rates
#   λ_z = γ z (1 - z/N)
#   μ_z = ν z
# ------------------------------
@inline function rates_sis(z::Int, gamma::Float64, nu::Float64, N::Int)
    λ = gamma * z * (1.0 - z / N)
    λ = max(λ, 0.0)
    μ = nu * z
    return λ, μ
end

@inline randexp_rate(rng::AbstractRNG, rate::Float64) = -log(rand(rng)) / rate

function simulate_path_exact(tmax::Float64, z0::Int, gamma::Float64, nu::Float64, N::Int, rng::AbstractRNG)
    t = 0.0
    z = z0
    times_evt  = Float64[0.0]
    states_evt = Int[z0]

    while t < tmax
        λ, μ = rates_sis(z, gamma, nu, N)
        total = λ + μ
        if total <= 0.0
            break
        end

        dt_jump = randexp_rate(rng, total)
        tnext = t + dt_jump
        if tnext > tmax
            break
        end

        if rand(rng) <= λ / total
            z = min(z + 1, N)
        else
            z = max(z - 1, 0)
        end

        t = tnext
        push!(times_evt, t)
        push!(states_evt, z)

        # extinction absorbing
        if z == 0
            break
        end
    end

    return times_evt, states_evt
end

function sample_on_grid(times_evt::Vector{Float64}, states_evt::Vector{Int}, grid::Vector{Float64})
    out = Vector{Float64}(undef, length(grid))
    j = 1
    for (i, tg) in pairs(grid)
        while j < length(times_evt) && times_evt[j+1] <= tg
            j += 1
        end
        out[i] = states_evt[j]
    end
    return out
end

function sample_theta(rng::AbstractRNG;
                      gamma_min::Float64, gamma_max::Float64,
                      nu_min::Float64, nu_max::Float64,
                      enforce_gamma_gt_nu::Bool)
    nu = rand(rng) * (nu_max - nu_min) + nu_min
    gamma = rand(rng) * (gamma_max - gamma_min) + gamma_min
    if enforce_gamma_gt_nu
        while gamma <= nu
            gamma = rand(rng) * (gamma_max - gamma_min) + gamma_min
        end
    end
    return gamma, nu
end

# Simulate ONE accepted sample (theta + Z) under mode constraint
function simulate_one_accepted(times::Vector{Float64}, z0::Int, N::Int, rng::AbstractRNG;
                               gamma_min::Float64, gamma_max::Float64,
                               nu_min::Float64, nu_max::Float64,
                               enforce_gamma_gt_nu::Bool,
                               m::Int,
                               mode::String)

    tmax = times[end]

    while true
        gamma, nu = sample_theta(rng;
            gamma_min=gamma_min, gamma_max=gamma_max,
            nu_min=nu_min, nu_max=nu_max,
            enforce_gamma_gt_nu=enforce_gamma_gt_nu
        )

        # build Zmat (T x m)
        Zmat = Matrix{Float64}(undef, length(times), m)
        extinct_flags = falses(m)

        for j in 1:m
            times_evt, states_evt = simulate_path_exact(tmax, z0, gamma, nu, N, rng)
            Zmat[:, j] .= sample_on_grid(times_evt, states_evt, times)
            extinct_flags[j] = (states_evt[end] == 0)  # extinction happened before/at tmax
        end

        # For m>1, we enforce based on ALL columns (keep it strict and simple)
        all_extinct     = all(extinct_flags)
        all_non_extinct = all(.!extinct_flags)

        if mode == "non_extinct" && all_non_extinct
            return gamma, nu, Zmat
        elseif mode == "extinct" && all_extinct
            return gamma, nu, Zmat
        end
        # else: reject and resample
    end
end

function build_split(K::Int, times::Vector{Float64}, z0::Int, N::Int, rng::AbstractRNG;
                     gamma_min::Float64, gamma_max::Float64,
                     nu_min::Float64, nu_max::Float64,
                     enforce_gamma_gt_nu::Bool,
                     m::Int,
                     mode::String)

    Z_list = Vector{Matrix{Float64}}(undef, K)
    theta  = Matrix{Float64}(undef, 2, K)  # [gamma; nu]

    for i in 1:K
        gamma, nu, Zmat = simulate_one_accepted(times, z0, N, rng;
            gamma_min=gamma_min, gamma_max=gamma_max,
            nu_min=nu_min, nu_max=nu_max,
            enforce_gamma_gt_nu=enforce_gamma_gt_nu,
            m=m,
            mode=mode
        )

        theta[1, i] = gamma
        theta[2, i] = nu
        Z_list[i] = Zmat
        
        # Show the simulating progress while running
        if i % 500 == 0
            println("  generated $i / $K ($mode)")
        end
    end

    return theta, Z_list
end

println("=== SIS DeepSets dataset generation ===")
println("mode=$mode | m=$m, dt=$dt, tmax=$tmax, T=$T")
println("K_train=$K_train, K_val=$K_val, K_test=$K_test")
println("Priors: gamma∼U($gamma_min,$gamma_max), nu∼U($nu_min,$nu_max)")
println("enforce_gamma_gt_nu = $enforce_gamma_gt_nu")
println("Saving to: $out_file\n")

theta_train, Z_train = build_split(K_train, times, z0, N, rng;
    gamma_min=gamma_min, gamma_max=gamma_max,
    nu_min=nu_min, nu_max=nu_max,
    enforce_gamma_gt_nu=enforce_gamma_gt_nu,
    m=m,
    mode=mode
)

theta_val, Z_val = build_split(K_val, times, z0, N, rng;
    gamma_min=gamma_min, gamma_max=gamma_max,
    nu_min=nu_min, nu_max=nu_max,
    enforce_gamma_gt_nu=enforce_gamma_gt_nu,
    m=m,
    mode=mode
)

theta_test, Z_test = build_split(K_test, times, z0, N, rng;
    gamma_min=gamma_min, gamma_max=gamma_max,
    nu_min=nu_min, nu_max=nu_max,
    enforce_gamma_gt_nu=enforce_gamma_gt_nu,
    m=m,
    mode=mode
)

@save out_file theta_train theta_val theta_test Z_train Z_val Z_test times T m dt tmax K_train K_val K_test N z0 gamma_min gamma_max nu_min nu_max enforce_gamma_gt_nu seed mode

println("Done.")
println("theta_train: ", size(theta_train), " | Z_train length: ", length(Z_train), " | each Z: ", size(Z_train[1]))
println("theta_val:   ", size(theta_val),   " | Z_val length:   ", length(Z_val))
println("theta_test:  ", size(theta_test),  " | Z_test length:  ", length(Z_test))