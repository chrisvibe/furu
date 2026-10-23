"""
furu pilot, the design-choice grid: every design factor on one temporal input stream (c = 1), with 12
task heads (density/parity x window {3, 5} x delay {0, 5, 10}) read from the same reservoir run, plus KQ/GR.
Config: config/temporal/multihead/grid_search/design_choices.yaml. Output: out/furu/grid_search/design_choices/.

Run: python -m project.boolean_reservoir.test.furu.generate_design_choices
Cluster: submit_job project.boolean_reservoir.test.furu.generate_design_choices --nodes <N> --no-tail
"""
from project.boolean_reservoir.code.train_model_parallel import boolean_reservoir_grid_search
from project.boolean_reservoir.code.utils.utils import run_on_nodes, configure_logging

configure_logging()

CONFIGS = ['config/temporal/multihead/grid_search/design_choices.yaml']


if __name__ == '__main__':
    run_on_nodes(CONFIGS, run_fn=boolean_reservoir_grid_search)
