from project.boolean_reservoir.code.utils.load_save import custom_load_grid_search_data
from project.boolean_reservoir.code.utils.metrics import get_reservoir_metrics, reservoir_key
from pathlib import Path
from project.boolean_reservoir.code.visualization import plot_grid_search

paths = [
    'config/temporal/density/grid_search/design_choices/all.yaml',
    'config/temporal/parity/grid_search/design_choices/all.yaml',
]

RESERVOIR_METRICS_EXTRACTION = ('M', lambda p: get_reservoir_metrics(p), {'lut_p'}, reservoir_key)

extractions = [
    ('T', lambda p: p.L.T, {'accuracy', 'loss'}),
    ('kqgr', lambda p: p.L.kqgr, {'kq', 'gr', 'delta'}),
    RESERVOIR_METRICS_EXTRACTION,
    ('L', lambda p: p.L, {'universe', 'out_path'}),
    ('L_out_name', lambda p: Path(p.L.out_path).name if p.L.out_path else None, None),
    ('kqgr', lambda p: p.U.kqgr.D, {'tau', 'tau_mode', 'tau_axis', 'evaluation'}),
    ('D', lambda p: p.D, {'task', 'window', 'delay'}),
    ('O', lambda p: p.M.O, {'mode'}),
    ('I', lambda p: p.M.I, {'perturbation', 'encoding', 'redundancy', 'chunks', 'ticks'}),
    ('R', lambda p: {
        **dict(p.M.R or {}),
        **{k[2:]: v for k, v in (p.M.variables or {}).items() if k.startswith('R_')},
    }, {'mode', 'k_avg', 'k_min', 'k_max', 'self_loops', 'n_nodes', 'init'}),
]

filter = lambda df: (
    (df['L_universe'] == 'train') &
    (df['I_ticks'] == '1') &
    (df['R_mode'] == 'heterogeneous') &
    (df['R_n_nodes'] == 512) &
    (df['R_self_loops'] == 0) &
    (df['R_k_avg'] == 3) &
    (df['D_task'] == 'density') &
    (df['kqgr_tau'] == 4) &
    True
)

df, cols = custom_load_grid_search_data(
    config_paths=paths,
    extractions=extractions,
    df_filter_mask=filter,
)


out_path = Path('/out/visualizations/grid_search/temporal/pca')
plot_grid_search(df, out_path)