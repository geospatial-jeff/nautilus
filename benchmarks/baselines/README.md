# Per-CPU throughput baselines

One file per CPU model, `<cpu-slug>.json`, each recorded on that hardware. Throughput is only comparable
on identical silicon, and GitHub-hosted runners land on a different physical CPU each run, so `bench-check`
gates a run against the baseline for *its own* CPU rather than one global number.

These files are grown automatically: on a push to `main`, a run that lands on a CPU with no baseline here
records one (`bench-check --bootstrap`) and the workflow opens a PR to add it. `../baseline.json` is the
seed — the pipeline set, scales, and the machine-independent digest reference that every run's output is
checked against, including a bootstrap on brand-new hardware.

Do not hand-edit these; regenerate with the `bench-baseline` workflow (which re-records the runner's CPU).
