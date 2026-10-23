"""
furu final run: every number in the paper comes from these grids. The choices were made on the pilot
(generate_design_choices); here they are scored on fresh reservoirs.
  design.yaml  the 7,560 designs x 20 seeds, 8 delayed targets on both readouts, KQ/GR under all 12 settings
  wiring.yaml  homogeneous vs heterogeneous at whole-number k; its heterogeneous half rebuilds design.yaml's
               reservoirs (same grid seed, seeding: content), so those rows must match design.yaml's
Output: out/furu/grid_search/final/{design,wiring}/.

Run: python -m project.boolean_reservoir.test.furu.generate_final
Cluster: submit_job project.boolean_reservoir.test.furu.generate_final --nodes <N> --no-tail
"""
from project.boolean_reservoir.code.train_model_parallel import boolean_reservoir_grid_search
from project.boolean_reservoir.code.utils.utils import run_on_nodes, configure_logging

configure_logging()

CONFIGS = ['config/temporal/multihead/final/design.yaml',
           'config/temporal/multihead/final/wiring.yaml']


if __name__ == '__main__':
    run_on_nodes(CONFIGS, run_fn=boolean_reservoir_grid_search)
