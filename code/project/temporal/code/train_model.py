from pathlib import Path
from project.boolean_reservoir.code.train_model import train_single_model
from project.temporal.code.visualization import plot_many_things
from project.boolean_reservoir.code.visualization import plot_activity_trace, plot_grid_search
from project.boolean_reservoir.code.train_model_parallel import boolean_reservoir_grid_search
from project.boolean_reservoir.code.utils.utils import run_on_nodes
from project.boolean_reservoir.code.utils.load_save import custom_load_grid_search_data, get_data_path
from project.boolean_reservoir.code.utils.metrics import get_reservoir_metrics, reservoir_key

import logging
from project.boolean_reservoir.code.utils.utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    pass 

    # # # debug 
    # # #####################################
    # from project.boolean_reservoir.code.reservoir import BooleanReservoir
    # from project.boolean_reservoir.code.utils import print_pretty_binary_matrix
    # import torch
    # from project.boolean_reservoir.code.parameter import load_yaml_config
    # from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
    # p = load_yaml_config('config/temporal/density/grid_search/homogeneous_deterministic.yaml')
    # p.L.out_path = '/out/debug'
    # p.L.history.record = True
    # p.L.save_keys = ['parameters', 'w_bi', 'graph', 'init_state', 'lut', 'weights']
    # p.M.I.connection = 'out-3:3:1'
    # p.M.I.n_nodes = 10
    # p.M.I.seed = p.M.R.seed = p.M.O.seed = 1
    # p.M.I.perturbation = 'override'
    # # p.M.I.perturbation = 'xor'
    # p.M.R.init = 'zeros'
    # p.M.R.n_nodes = 30
    # p.M.R.k_avg = 4
    # configs = generate_param_combinations(p)
    # model = BooleanReservoir(configs[0])
    # # model = BooleanReservoir(load_path='/out/test/temporal/density/grid_search/homogeneous-deterministic/runs/2025_06_25_125038_810586/checkpoints/last_checkpoint')
    # x = torch.tensor([[[[int(bit)]] for bit in '1001001010']], dtype=torch.uint8)
    # # model(x)
    # # model.save()
    # # model.flush_history()
    # p, model, dataset, history = train_single_model(model=model, dataset_init=d(), accuracy=a().accuracy)
    # plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0])
    # pass


    # # these just add to the grid search below a 50% model and a 100% model
    # p, model, dataset, history = train_single_model('/out/test/temporal/density/grid_search/homogeneous-deterministic/runs/2025_06_25_125038_810586/checkpoints/last_checkpoint/parameters.yaml', dataset_init=d(), accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)
    # p, model, dataset, history = train_single_model('/out/test/temporal/density/grid_search/homogeneous-deterministic/runs/2025_06_25_125107_553781/checkpoints/last_checkpoint/parameters.yaml', dataset_init=d(), accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)

    # boolean_reservoir_grid_search(
    #     'config/temporal/density/test/debug_homogeneous_deterministic.yaml',
    #     dataset_init=d(),
    #     accuracy=a().accuracy,
    #     gpu_memory_per_job_gb = 1/2,
    #     cpu_memory_per_job_gb = 1/2,
    #     cpu_cores_per_job = 1,
    # )

    # # # Simple run
    # # #####################################

    # p, model, dataset, history = train_single_model('config/temporal/density/single_run/ok_model.yaml')
    # plot_many_things(model, dataset, history)
    # plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0])

    # p, model, dataset, history = train_single_model('config/temporal/density/single_run/sample_model.yaml', ignore_gpu=True)
    # model.save()

    # # plot_many_things(model, dataset, history)
    # plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0])

    # config = '/out/delete/params.yaml'
    # boolean_reservoir_grid_search(config)
    # extractions = [
    # ('T', lambda p: p.L.T, {'accuracy', 'loss'}),
    # ('R', lambda p: {
    #     **dict(p.M.R or {}),
    #     **{k[2:]: v for k, v in (p.M.variables or {}).items() if k.startswith('R_')},
    # }, {'mode', 'k_avg', 'k_min', 'k_max', 'self_loops', 'n_nodes', 'init'}),
    # ]
    # df, factors = custom_load_grid_search_data(config_paths=[config], extractions=extractions)
    # pass