# NBE for BD Process

Estimators for the rate parameters of population birth-death models, from
simulated trajectories:

- **LBDP** — linear birth-death process, rates `(lambda, mu)`.
- **LBDP (Q-process, subcritical)** — LBDP conditioned to survive, rates `(lambda, mu)` with `lambda < mu`.
- **SIS / Verhulst** — logistic (density-dependent) birth-death process, rates `(gamma, nu)`.

For each model there's a simulator (generates train/val/test trajectories) and
one or more estimators that recover the rate parameters from a trajectory:
a 1D-CNN neural Bayes estimator, a two-network DeepSets ("SR") neural
estimator, a classical GW moment estimator, and a consistent least-squares
estimator (SIS only).

## Setup

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run everything below from the repo root so relative paths (`data/...`) resolve correctly.

## Layout

```
simulators/     generate train/val/test trajectories -> data/*.jld2
estimators/     fit a parameter estimator against a dataset in data/
plotting/       (Python) plot results exported from the estimators
data/           generated datasets, git-ignored
```

## Pipeline

### LBDP (continuous-time SSA)

```
julia --project=. simulators/lbdp_ssa.jl non_extinct      # -> data/lbdp_non_extinct.jld2
julia --project=. estimators/nbe_cnn.jl non_extinct
julia --project=. estimators/sr_deepsets.jl non_extinct
julia --project=. estimators/gw_moment.jl data/lbdp_non_extinct.jld2
```

Replace `non_extinct` with `extinct` to allow extinction instead of rejecting for it.
Optional simulator args: `<mode> <M> <dt> <tmax> <K_train> <K_val> <K_test> <r_max>`.

### LBDP (discrete-time, Binomial/NegBinomial transition)

An alternative to the SSA simulator above, same prior/model, different
simulation method — kept separate so the two don't overwrite each other's data:

```
julia --project=. simulators/lbdp_discrete.jl non_extinct  # -> data/lbdp_discrete_non_extinct.jld2
```

### LBDP, subcritical only (Q-process, non-extinct by construction)

```
julia --project=. simulators/lbdp_q_subcritical.jl         # -> data/lbdp_q_subcritical_non_extinct.jld2
julia --project=. estimators/nbe_cnn.jl non_extinct_Q
julia --project=. estimators/sr_deepsets.jl non_extinct_Q
```

### SIS / Verhulst

```
julia --project=. simulators/sis.jl non_extinct            # -> data/sis_non_extinct.jld2
julia --project=. estimators/nbe_cnn.jl non_extinct_SIS
julia --project=. estimators/sis_consistent_ls.jl
```

### Plotting

`plotting/plot_subcritical_results.py` is a standalone Python/matplotlib script
with hardcoded MAE-vs-observations numbers copied from prior runs of the SR and
CNN estimators on the subcritical case; update the `results` dict with new
numbers before running it.
