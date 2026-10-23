# Boolean reservoirs on binary bit streams

Code and data for the paper: random Boolean network reservoirs with a linear readout, on temporal density
and parity tasks. CPU only.

## Setup

```bash
mamba env create -f docker/src/environment.lock.yaml   # the exact versions behind the paper (Linux)
conda activate boolean_reservoir
```

The reservoir runs on Julia: the first import downloads and compiles it (a few minutes, needs internet).
`docker/Dockerfile` builds the same environment as a container.

## Reproduce

From `code/`. Every number in the paper comes from the per-reservoir results in `out/furu/tables/`
(about 30 s, 4 GB of memory):

```bash
python -m project.boolean_reservoir.test.furu.analyze_design_choices   # -> out/furu/grid_search/final/design/analysis/
```

The script's docstring maps each output table to its result. To regenerate the results instead (about 6,000 core-hours):

```bash
python -m project.boolean_reservoir.test.furu.generate_design_choices  # the run the choices were made on
python -m project.boolean_reservoir.test.furu.generate_final           # the reported run
python -m project.boolean_reservoir.test.furu.generate_replication     # Snyder et al. (2012), figure 1
```

The first run's probe settings were re-measured by a script not included here, so the analysis always reads
that run from `out/furu/tables/pilot/`. The raw grid output (every job's parameters and fit diagnostics, about
900 MB) is not included; it is available from the authors on request.

Tests: `python -m pytest` from `code/`.

## Figures

`DASHBOARD_SITE=furu python dashboard/run.py <app>` with `design`, `polar`, `analysis` or `figure1`, then open
http://localhost:8050. Run the analysis first; `figure1` needs the replication run.

## License

MIT (see `LICENSE`).
