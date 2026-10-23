def compute_reservoir_metrics(p) -> dict:
    from project.boolean_reservoir.code.reservoir import BooleanReservoir
    from project.boolean_reservoir.code.lut import calc_lut_p
    params = p.to_pydantic()
    model = BooleanReservoir.build_wiring_and_lut(params)
    return {
        # "spectral_radius": float(calc_spectral_radius(model.graph)),
        "lut_p": calc_lut_p(model.lut),
    }

def _stable(obj):
    """Plain, order-stable form of a DotDict / pydantic model / scalar."""
    if hasattr(obj, 'model_dump'):
        return _stable(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _stable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_stable(v) for v in obj]
    return obj


def reservoir_key(p):
    """Key on everything build_wiring_and_lut actually reads.

    Was (R.seed, R.n_nodes). That is not what determines the reservoir: k_avg,
    k_min, k_max, self_loops, p and the wiring type all change the graph and the
    LUT. Configs differing only in those therefore collided and shared one cached
    lut_p — silently wrong values in the dashboard (wsgi/base.py). Wiring order is
    significant, so lists keep their order; dicts are sorted for stability.
    """
    import json
    M = p.M
    return (
        M.R.seed, M.R.n_nodes, M.R.p,
        M.I.n_nodes, M.I.bits, M.O.n_nodes,
        json.dumps(_stable(getattr(M, 'wiring', None)), default=str),
        json.dumps(_stable(getattr(M, 'variables', None) or {}), default=str),
    )

def get_reservoir_metrics(p) -> dict:
    """Return stored reservoir metrics if available, else compute them.

    Uses stored metrics if lut_p is present. spectral_radius is intentionally
    null in all current parquets and is excluded from the availability check.
    """
    rm = getattr(p.L, 'reservoir_metrics', None)
    if rm is not None:
        stored = rm.model_dump() if hasattr(rm, 'model_dump') else dict(rm)
        if stored.get('lut_p') is not None:
            return stored
    return compute_reservoir_metrics(p)
