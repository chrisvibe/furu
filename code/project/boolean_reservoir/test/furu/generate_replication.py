"""
Regenerate the KQ/GR replication of Snyder et al. (2012, figure 1) behind the paper's
short driven-vs-autonomous part: KQ/GR against connectivity under five input-wiring
themes, from the shared (I+R)->R pool, where the input can stay unconnected, to a
dedicated I->R mapping. KQ/GR only, no readout training; small (25- and 128-node
reservoirs), so it runs on a workstation.

Run: python -m project.boolean_reservoir.test.furu.generate_replication
"""
from project.boolean_reservoir.code.train_model_parallel import boolean_reservoir_grid_search
from project.boolean_reservoir.code.utils.utils import run_on_nodes, configure_logging

configure_logging()

CONFIGS = [
    'config/temporal/kqgr/grid_search/figure_1_snyder_2012.yaml',
]


if __name__ == '__main__':
    run_on_nodes(CONFIGS, run_fn=boolean_reservoir_grid_search)
