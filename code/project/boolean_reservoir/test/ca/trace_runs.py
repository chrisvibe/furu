"""The CA trace runs the trace dashboard shows: one entry per CA config, each writing its
runs (history + checkpoint) under that config's out_path. The tests prove each rule
correct against a reference; these runs show what it looks like.

Run them all with test/hegg/generate_ca_trace_runs.py.
"""
from pathlib import Path

import numpy as np

from project.boolean_reservoir.code import ca
from project.boolean_reservoir.code.parameter import InputParams

CONFIG_DIR = Path('config') / 'test' / 'boolean_reservoir' / 'ca'


def record(model, x, file_name: str) -> None:
    """One run in its own run directory: forward x, write history + checkpoint, plot."""
    model.init_logging()
    model.eval()
    model(x)
    model.flush_history()
    model.save()
    try:
        from project.boolean_reservoir.code.visualization import plot_activity_trace
        plot_activity_trace(model.save_path, file_name=file_name)
    except Exception as e:  # the dashboard reads the history; the image is a convenience
        print(f'  (no {file_name}: {type(e).__name__}: {e})')
    print(f'  {file_name}: {model.save_path}')


def rule90():
    """Sierpinski triangle from a single seeded centre cell. Left-right symmetric: Rule 90
    is the XOR of two equidistant neighbours and the seed sits in the middle."""
    model = ca.ca_model(CONFIG_DIR / 'rule90.yaml', ca.install_rule90,
                        edit=lambda P: setattr(P.L.history, 'persist_to_disk', True))
    seed = np.zeros(model.R.n_nodes, dtype=np.uint8)
    seed[model.R.n_nodes // 2] = 1
    ca.set_state(model, seed)
    record(model, ca.ca_input(np.zeros(32, dtype=np.uint8), model.P), 'rule90_sierpinski.svg')


def _rule150(config: str, injected: str, dual: bool):
    T = 64
    impulse = np.zeros(T, dtype=np.uint8); impulse[0] = 1
    random_bits = np.random.default_rng(0).integers(0, 2, T, dtype=np.uint8)
    install = lambda m: ca.install_rule150(m, injected)
    stem = Path(config).stem

    # Impulse: one activation at cell 0 spreads right as a diagonal stripe.
    model = ca.ca_model(CONFIG_DIR / config, install, viz=True)
    record(model, ca.ca_input(impulse, model.P), f'{stem}_impulse.svg')
    if not dual:
        return
    # Random stream: each new bit interferes with the texture already in the ring.
    record(model, ca.ca_input(random_bits, model.P), f'{stem}_random.svg')

    # Redundancy 2: the same bit enters at cell 0 and cell N/2, two wavefronts that meet
    # mid-ring. The random run is symmetric around N/2: both sites get the same stream.
    def two_injection_points(P):
        P.M.input_layer = InputParams.model_validate({
            **P.M.I.model_dump(), 'redundancy': 2,
            'bits': None, 'n_nodes': None, 'chunk_size': None})  # recomputed from redundancy
    model = ca.ca_model(CONFIG_DIR / config, install, viz=True, edit=two_injection_points)
    record(model, ca.ca_input(impulse, model.P), f'{stem}_dual_impulse.svg')
    record(model, ca.ca_input(random_bits, model.P), f'{stem}_dual_random.svg')


def rule150_transducer():
    _rule150('rule150_transducer.yaml', 'xor', dual=True)


def rule150_linear_transducer():
    """Cell 0 is the input itself: a sharper left edge than the transducer, no memory there."""
    _rule150('rule150_linear_transducer.yaml', 'input', dual=False)


def gkl_density():
    """Density classification on a 64-cell ring: ρ=0.8 → all 1s, ρ=0.2 → all 0s, and a
    two-domain start (ρ=0.5) that shows the domain-wall dynamics."""
    def model():
        return ca.ca_model(CONFIG_DIR / 'gkl_density.yaml', ca.install_gkl, viz=True,
                           edit=lambda P: setattr(P.M.R, 'n_nodes', 64))
    T_CONV, T_WALL = 3 * 149, 128
    for name, density in (('gkl_high_density.svg', 0.8), ('gkl_low_density.svg', 0.2)):
        m = model()
        ca.set_state(m, (np.random.default_rng(0).random(m.R.n_nodes) < density).astype(np.uint8))
        record(m, ca.ca_input(np.zeros(T_CONV, dtype=np.uint8), m.P), name)
    m = model()
    ca.set_state(m, (np.arange(m.R.n_nodes) >= m.R.n_nodes // 2).astype(np.uint8))
    record(m, ca.ca_input(np.zeros(T_WALL, dtype=np.uint8), m.P), 'gkl_boundary.svg')


def second_order_rule90():
    """Three keyed lock/unlock round trips (key seeds 0, 1, 2): T forward steps lock the
    seeded state, T backward steps with the reversed key unlock it, in one history."""
    T = 48
    for key_seed in (0, 1, 2):
        model = ca.ca_model(CONFIG_DIR / 'second_order_rule90.yaml', ca.install_second_order_rule90,
                            edit=lambda P: setattr(P.L.history, 'persist_to_disk', True))
        n = model.R.n_nodes // 2
        key = np.random.default_rng(key_seed).integers(0, 2, T, dtype=np.uint8)
        x0 = np.zeros(n, dtype=np.uint8); x0[n // 2] = 1
        ca.set_state(model, x0)
        model.eval()
        model(ca.ca_input(key, model.P))
        model.flush_history()

        locked = model.states_parallel[0, model.res_slice].copy()
        ca.install_second_order_rule90(model, 'backward')
        ca.set_state(model, locked)
        model(ca.ca_input(key[::-1].copy(), model.P))
        model.flush_history()
        model.save()
        print(f'  key_seed={key_seed}: {model.save_path}')


RUNS = {f.__name__: f for f in (rule90, rule150_transducer, rule150_linear_transducer,
                                gkl_density, second_order_rule90)}  # name = config stem
