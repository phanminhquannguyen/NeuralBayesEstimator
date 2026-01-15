# data_summary.jl
# Detailed summary for LBDP NBE datasets (extinction / survival by regime)
#
# Usage:
#   julia data_summary.jl nbe_lbdp_extinct.jld2
#   julia data_summary.jl nbe_lbdp_non_extinct.jld2

using JLD2
using Statistics
using Printf

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

# replicate extinct if Z(T) == 0
extinct_mask(Zk::AbstractMatrix) = Zk[end, :] .== 0
extinct_count(Zk::AbstractMatrix) = count(extinct_mask(Zk))
survive_count(Zk::AbstractMatrix) = size(Zk, 2) - extinct_count(Zk)

# dataset-level indicators (treat each dataset = one theta with m replicates)
dataset_all_extinct(Zk) = extinct_count(Zk) == size(Zk, 2)
dataset_all_survive(Zk) = extinct_count(Zk) == 0
dataset_some_survive(Zk) = !dataset_all_extinct(Zk)  # at least 1 survivor replicate


function summarize_split(name, theta, Z_list)
    K = size(theta, 2)
    T, m = size(Z_list[1])

    lambda = theta[1, :]
    mu     = theta[2, :]

    is_sub   = lambda .< mu
    is_super = lambda .> mu
    is_crit  = lambda .== mu

    # extinct replicates per dataset (length K)
    ext_counts = [extinct_count(Zk) for Zk in Z_list]

    # ---- totals ----
    total_datasets     = K
    total_trajectories = K * m
    total_extinct      = sum(ext_counts)
    total_survive      = total_trajectories - total_extinct

    println("[$name]")
    @printf("  Total datasets (theta draws): %d\n", total_datasets)
    @printf("  Replicates per dataset:       %d\n", m)
    @printf("  Total trajectories:           %d\n", total_trajectories)
    @printf("  Overall extinct trajectories: %d (%.2f%%)\n",
            total_extinct, 100 * total_extinct / total_trajectories)
    @printf("  Overall survive trajectories: %d (%.2f%%)\n",
            total_survive, 100 * total_survive / total_trajectories)
    println()

    # ---- helper: regime summary in TRAJECTORY space ----
    function traj_regime_report(label, mask)
        K_r = count(mask)
        traj_r = K_r * m

        if traj_r == 0
            @printf("  %s: none\n\n", label)
            return
        end

        ext_r = sum(ext_counts[mask])        # extinct trajectories in regime
        surv_r = traj_r - ext_r              # surviving trajectories in regime

        @printf("  %s\n", label)
        @printf("    datasets (theta):      %d (%.2f%% of datasets)\n",
                K_r, 100 * K_r / K)
        @printf("    trajectories total:    %d (%.2f%% of all trajectories)\n",
                traj_r, 100 * traj_r / total_trajectories)
        @printf("    trajectories extinct:  %d (%.2f%% within regime)\n",
                ext_r, 100 * ext_r / traj_r)
        @printf("    trajectories survive:  %d (%.2f%% within regime)\n",
                surv_r, 100 * surv_r / traj_r)
        println()
    end

    traj_regime_report("Subcritical (lambda < mu)", is_sub)
    traj_regime_report("Supercritical (lambda > mu)", is_super)
    traj_regime_report("Critical (lambda = mu)", is_crit)

    println("  (Each trajectory is counted under the regime of its dataset's theta.)")
    println()
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

function main()
    if length(ARGS) != 1
        println("Usage: julia data_summary.jl <datafile>")
        println("Example: julia data_summary.jl nbe_lbdp_extinct.jld2")
        return
    end

    datafile = ARGS[1]

    @load datafile theta_train theta_val theta_test Z_train Z_val Z_test times m T r_max

    println("============================================================")
    println("DATA SUMMARY: ", datafile)
    println("============================================================")

    @printf("Time grid: T=%d, t_min=%.3f, t_max=%.3f\n", T, times[1], times[end])
    @printf("Replicates per dataset: m=%d\n", m)
    @printf("Prior: lambda, mu ~ Uniform(0, %.2f)\n", r_max)
    println()

    summarize_split("TRAIN", theta_train, Z_train)
    summarize_split("VALIDATION", theta_val, Z_val)
    summarize_split("TEST", theta_test, Z_test)
end

main()