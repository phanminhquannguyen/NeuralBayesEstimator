# ============================================================
# run_pipeline_T.jl
# Sweep over different T (number of observation times), run:
#   1) simulate data
#   2) train model
#   3) evaluate model (writes outputs/test_metrics_by_regime.csv)
# Then aggregate "Test / All" metrics and make a line plot vs T.
#
# Assumes you already have these scripts (with matching CLI args):
#   simulate_dataset.jl  mode out_data_file m dt tmax K_train K_val K_test r_max
#   train_nbe.jl         mode data_file model_file
#   evaluate_nbe.jl      mode model_file data_file out_dir
#
# Run:
#   julia run_pipeline_T.jl
# ============================================================

using Dates
using Printf
using DataFrames
using CSV
using Plots

# -----------------------------
# User settings (edit here)
# -----------------------------
mode   = "extinct"          # "extinct" or "non_extinct"
m      = 20                # number of trajectories per dataset
tmax   = 1.9               # horizon; dt is chosen from T so that (T-1)*dt = tmax
T_grid = [10, 20, 40, 80]  # sweep T first

K_train = 10_000
K_val   = 2_000
K_test  = 2_000
r_max   = 5.0

ROOT = "sweeps"
mkpath(ROOT)

# -----------------------------
# Helper: run a command (throws on error)
# -----------------------------
function run_cmd(cmd::Cmd)
    println("\n> ", cmd)
    run(cmd)
end

function main()
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    sweep_dir = joinpath(ROOT, "sweep_T_" * mode * "_" * stamp)
    mkpath(sweep_dir)

    println("Sweep dir: ", sweep_dir)

    results = DataFrame(
        T = Int[],
        dt = Float64[],
        BiasLambda = Float64[],
        RMSELambda = Float64[],
        BiasMu = Float64[],
        RMSEMu = Float64[],
        MeanAbsBias = Float64[],
        MeanRMSE = Float64[]
    )

    for T in T_grid
        dt = tmax / (T - 1)  # guarantees exactly T points on 0:dt:tmax

        tag = @sprintf("T%d_m%d_tmax%.1f", T, m, tmax)
        run_dir = joinpath(sweep_dir, tag)
        out_dir = joinpath(run_dir, "outputs")
        mkpath(out_dir)

        data_file  = joinpath(run_dir, "data_" * mode * ".jld2")
        model_file = joinpath(run_dir, "trained_" * mode * ".jld2")

        println("\n===================================================")
        println("RUN: ", tag, "   mode=", mode, "   dt=", dt)
        println("===================================================")

        # 1) simulate
        run_cmd(`julia simulator.jl $mode $data_file $m $dt $tmax $K_train $K_val $K_test $r_max`)

        # 2) train
        run_cmd(`julia NBE_training.jl $mode $data_file $model_file`)

        # 3) evaluate (writes outputs/test_metrics_by_regime.csv)
        run_cmd(`julia evaluate_nbe.jl $mode $model_file $data_file $out_dir`)

        # read metrics and extract "Test / All"
        metrics_csv = joinpath(out_dir, "test_metrics_by_regime.csv")
        df = CSV.read(metrics_csv, DataFrame)

        row = df[(df.Split .== "Test") .& (df.Regime .== "All"), :]
        if nrow(row) != 1
            error("Could not find exactly one row for Split=Test, Regime=All in $metrics_csv")
        end

        bλ = row.BiasLambda[1]
        rλ = row.RMSELambda[1]
        bμ = row.BiasMu[1]
        rμ = row.RMSEMu[1]

        mean_abs_bias = (abs(bλ) + abs(bμ)) / 2
        mean_rmse     = (rλ + rμ) / 2

        push!(results, (T, dt, bλ, rλ, bμ, rμ, mean_abs_bias, mean_rmse))
    end

    # save summary
    summary_csv = joinpath(sweep_dir, "summary_bias_rmse_vs_T.csv")
    CSV.write(summary_csv, results)
    println("\nSaved summary table: ", summary_csv)
    println(results)

    # -----------------------------
    # Plot: MeanAbsBias & MeanRMSE vs T
    # -----------------------------
    p = plot(
        results.T, results.MeanAbsBias;
        xlabel = "T (number of observation times)",
        ylabel = "Metric",
        label  = "Mean |Bias| (avg over λ, μ)",
        marker = :circle
    )
    plot!(p, results.T, results.MeanRMSE; label="Mean RMSE (avg over λ, μ)", marker=:circle)

    plot_path = joinpath(sweep_dir, "lineplot_bias_rmse_vs_T.pdf")
    savefig(p, plot_path)
    println("\nSaved line plot: ", plot_path)

    println("\nDone. All results under: ", sweep_dir)
end

main()