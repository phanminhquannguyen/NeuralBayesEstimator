# ============================================================
# common.jl
# Shared helpers used by the LBDP/Q-process simulators.
# ============================================================

using Random
using Distributions

# Map an event-based trajectory (event_times, event_states) onto a fixed
# observation grid: for each grid time, take the last event state at or
# before it.
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

# Prior sampler: Theta is 2xK with rows (lambda, mu) ~ Uniform(1, r_max) each,
# rejection-sampled so that cmp(lambda, mu) holds (cmp = > for
# critical/supercritical-only, cmp = < for subcritical-only) and
# abs(lambda - mu) <= delta. Pass `rng` explicitly to match a caller that
# seeds its own RNG object rather than the global one.
function sample_theta_constrained(K::Int; r_max::Real, delta::Real, cmp::Function,
                                  rng::AbstractRNG = Random.default_rng())
    theta = Matrix{Float64}(undef, 2, K)
    k = 1
    while k <= K
        lambda = rand(rng, Uniform(1, r_max))
        mu = rand(rng, Uniform(1, r_max))
        if cmp(lambda, mu) && abs(lambda - mu) <= delta
            theta[1, k] = lambda
            theta[2, k] = mu
            k += 1
        end
    end
    return theta
end
