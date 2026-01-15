# ============================================================
# evaluate_nbe.jl
# Evaluate a saved DeepSets PointEstimator on test data
# + save metrics + plots (PDF) for either extinct or non_extinct case.
#
# Run:
#   julia evaluate_nbe.jl extinct
#   julia evaluate_nbe.jl non_extinct
#
# Optional:
#   julia evaluate_nbe.jl <extinct|non_extinct> <model_file.jld2> <test_data_file.jld2> <output_dir>
# ============================================================

using JLD2
using Statistics
using Printf

using NeuralEstimators
using Flux

using Plots
using StatsPlots

using DataFrames
using CSV


# Helpers
subset_theta(theta::AbstractMatrix, mask) = theta[:, mask]
subset_Z(Z_list::AbstractVector, mask) = Z_list[findall(mask)]

function metrics_row(estimator, theta_true::AbstractMatrix, Z_list::AbstractVector, split::String, regime::String)
    a = assess(estimator, theta_true, Z_list; use_gpu=false)
    b = bias(a)
    r = rmse(a)
    return (split, regime, b.bias[1], r.rmse[1], b.bias[2], r.rmse[2])
end


function main()
    if length(ARGS) < 1 || !(ARGS[1] in ["extinct", "non_extinct"])
        println("Usage: julia evaluate_nbe.jl <extinct|non_extinct> [model_file] [test_file] [out_dir]")
        exit(1)
    end

    case = ARGS[1]

    default_model = case == "extinct" ? "trained_estimator_extinct.jld2" : "trained_estimator_non_extinct.jld2"
    default_test  = "test_data.jld2"
    default_out   = "outputs"

    model_file = length(ARGS) >= 2 ? ARGS[2] : default_model
    test_file  = length(ARGS) >= 3 ? ARGS[3] : default_test
    out_dir    = length(ARGS) >= 4 ? ARGS[4] : default_out

    # case-specific output folder
    out_dir = joinpath(out_dir, case)
    mkpath(out_dir)

    println("============================================================")
    println("Case:      ", case)
    println("Model:     ", model_file)
    println("Test data: ", test_file)
    println("Outputs:   ", out_dir)
    println("============================================================")

    # -----------------------------
    # 1) Load model
    # -----------------------------
    @load model_file estimator
    println("Loaded estimator.")

    # -----------------------------
    # 2) Load test data
    # -----------------------------
    @load test_file theta_test Z_test
    println("Loaded test data.")
    println("  theta_test size: ", size(theta_test))
    println("  Z_test length:   ", length(Z_test), " datasets; each size ", size(Z_test[1]))

    # -----------------------------
    # 3) Convert to Float64
    # -----------------------------
    theta_test64 = Float64.(theta_test)
    Z_test64     = [Float64.(Zk) for Zk in Z_test]

    lambda_true = theta_test64[1, :]
    mu_true     = theta_test64[2, :]

    mask_sub   = lambda_true .< mu_true
    mask_super = lambda_true .> mu_true
    mask_crit  = lambda_true .== mu_true

    # -----------------------------
    # 4) Metrics table (overall + by regime)
    # -----------------------------
    rows = DataFrame(
        Split = String[],
        Regime = String[],
        BiasLambda = Float64[],
        RMSELambda = Float64[],
        BiasMu = Float64[],
        RMSEMu = Float64[]
    )

    # overall
    split, regime, bl, rl, bm, rm = metrics_row(estimator, theta_test64, Z_test64, "Test", "All")
    push!(rows, (split, regime, bl, rl, bm, rm))

    # subcritical
    if any(mask_sub)
        θ_sub = subset_theta(theta_test64, mask_sub)
        Z_sub = subset_Z(Z_test64, mask_sub)
        split, regime, bl, rl, bm, rm = metrics_row(estimator, θ_sub, Z_sub, "Test", "Subcritical")
        push!(rows, (split, regime, bl, rl, bm, rm))
    end

    # supercritical
    if any(mask_super)
        θ_sup = subset_theta(theta_test64, mask_super)
        Z_sup = subset_Z(Z_test64, mask_super)
        split, regime, bl, rl, bm, rm = metrics_row(estimator, θ_sup, Z_sup, "Test", "Supercritical")
        push!(rows, (split, regime, bl, rl, bm, rm))
    end

    # critical (rare)
    if any(mask_crit)
        θ_c = subset_theta(theta_test64, mask_crit)
        Z_c = subset_Z(Z_test64, mask_crit)
        split, regime, bl, rl, bm, rm = metrics_row(estimator, θ_c, Z_c, "Test", "Critical")
        push!(rows, (split, regime, bl, rl, bm, rm))
    end

    metrics_csv = joinpath(out_dir, "test_metrics_by_regime.csv")
    CSV.write(metrics_csv, rows)
    println("\nSaved metrics table: ", metrics_csv)
    println(rows)

    # -----------------------------
    # 5) Predict once for plots
    # -----------------------------
    theta_hat = estimate(estimator, Z_test64; use_gpu=false)  # 2×K

    # -----------------------------
    # 6) Scatter: true vs predicted (λ and μ) side-by-side
    # -----------------------------
    λmin, λmax = minimum(lambda_true), maximum(lambda_true)
    μmin, μmax = minimum(mu_true), maximum(mu_true)

    p_lambda = scatter(
        lambda_true, theta_hat[1, :],
        xlabel="True λ", ylabel="Predicted λ",
        title="True vs Predicted λ",
        markersize=2, legend=false
    )
    plot!(p_lambda, [λmin, λmax], [λmin, λmax], color=:red, linewidth=3, linestyle=:solid)

    p_mu = scatter(
        mu_true, theta_hat[2, :],
        xlabel="True μ", ylabel="Predicted μ",
        title="True vs Predicted μ",
        markersize=2, legend=false
    )
    plot!(p_mu, [μmin, μmax], [μmin, μmax], color=:red, linewidth=3, linestyle=:solid)

    p_scatter = plot(p_lambda, p_mu, layout=(1,2), size=(1000, 420))

    scatter_pdf = joinpath(out_dir, "true_vs_pred_lambda_mu.pdf")
    savefig(p_scatter, scatter_pdf)
    println("Saved scatter plot: ", scatter_pdf)

    # -----------------------------
    # 7) Boxplot: absolute errors (4 boxes with gaps)
    #    Super λ, Super μ, Sub λ, Sub μ
    # -----------------------------
    abs_err_lambda = abs.(theta_hat[1, :] .- lambda_true)
    abs_err_mu     = abs.(theta_hat[2, :] .- mu_true)

    err_super_lambda = abs_err_lambda[mask_super]
    err_super_mu     = abs_err_mu[mask_super]
    err_sub_lambda   = abs_err_lambda[mask_sub]
    err_sub_mu       = abs_err_mu[mask_sub]

    x = vcat(
        fill(0.0, length(err_super_lambda)),
        fill(2.0, length(err_super_mu)),
        fill(4.0, length(err_sub_lambda)),
        fill(6.0, length(err_sub_mu))
    )

    y = vcat(err_super_lambda, err_super_mu, err_sub_lambda, err_sub_mu)

    p_box = boxplot(
        x, y;
        boxwidth = 0.35,
        legend = false,
        ylabel = "|error|",
        title = "Absolute Error by Regime and Parameter",
        xticks = ([0, 2, 4, 6], ["Super λ", "Super μ", "Sub λ", "Sub μ"]),
        xlims = (-1, 7),
        size = (900, 420)
    )

    box_pdf = joinpath(out_dir, "boxplot_error_by_regime.pdf")
    savefig(p_box, box_pdf)
    println("Saved boxplot: ", box_pdf)

    println("\nDone.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end