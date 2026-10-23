"""pytest tests for `apply_figure_options`: the per-view publication options in a view's
_extras['figure'] (paper notation, not config keys).
"""
import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # code/dashboard, as the other tests

from dashboards.utils import apply_figure_options  # noqa: E402


def _ladder(dots: bool = True):
    """The ladder's shape: per series a markers trace and its fitted line, under the generated name.
    dots=False: as the scatter dashboard builds it with dots off and the markers hidden (trace_visibility)."""
    fig = go.Figure(layout={'title': {'text': 'ladder'}})
    hidden = {} if dots else {'showlegend': False, 'visible': False}
    for name, dash in (('linear [log10_nmse]', 'solid'), ('query [log10_nmse]', 'solid'),
                       ('query [log10_nmse_steps]', 'dash'), ('linear [log10_nmse_steps]', 'dash')):
        fig.add_scatter(x=[1, 2], y=[0, 1], mode='markers', name=name, legendgroup=name, **hidden)
        fig.add_scatter(x=[1, 2], y=[0, 1], mode='lines', name=name, line_dash=dash)
    return fig


NAMES = {'linear [log10_nmse]': 'integration', 'query [log10_nmse_steps]': 'retention',
         'query [log10_nmse]': 'selection', 'linear [log10_nmse_steps]': None}


def _legend(fig):
    """The legend as a reader sees it: (rank, name, mode) of each trace that shows an entry."""
    return sorted((t.legendrank, t.name, t.mode) for t in fig.data if t.showlegend is not False and t.visible is not False)


def test_no_options_leaves_figure_alone():
    fig = _ladder()
    before = fig.to_dict()
    apply_figure_options(fig, None)
    assert fig.to_dict() == before


def test_names_rename_order_and_drop():
    fig = _ladder()
    apply_figure_options(fig, {'names': NAMES})
    names = {t.name for t in fig.data}
    assert names == {'integration', 'retention', 'selection', 'linear [log10_nmse_steps]'}
    dropped = [t for t in fig.data if t.name == 'linear [log10_nmse_steps]']
    assert all(t.showlegend is False for t in dropped)
    assert [n for _, n, _ in _legend(fig)][::2] == ['integration', 'retention', 'selection']


def test_unlisted_series_keep_their_name():
    fig = _ladder()
    apply_figure_options(fig, {'names': {'linear [log10_nmse]': 'integration'}})
    assert {t.name for t in fig.data} >= {'integration', 'query [log10_nmse]'}


def test_per_series_markers_carry_the_entry():
    fig = _ladder()
    apply_figure_options(fig, {'legend': 'per-series', 'names': NAMES})
    assert _legend(fig) == [(0, 'integration', 'markers'), (1, 'retention', 'markers'), (2, 'selection', 'markers')]
    for t in fig.data:
        if t.mode == 'lines':
            assert t.legendgroup == next(m.legendgroup for m in fig.data if m.mode == 'markers' and m.name == t.name)


def test_line_swatch_moves_the_entry_to_the_fitted_line():
    """retention and selection share a colour; only the line's dash tells them apart, so the entry must be the line."""
    fig = _ladder()
    apply_figure_options(fig, {'legend': 'per-series', 'legend_swatch': 'line', 'names': NAMES})
    assert _legend(fig) == [(0, 'integration', 'lines'), (1, 'retention', 'lines'), (2, 'selection', 'lines')]


def test_dots_off_keeps_one_entry_per_line():
    """Dots off (markers without an entry, hidden): each fitted line keeps its entry, with or without the line swatch
    (the legend once vanished)."""
    for options in ({'legend': 'per-series'}, {'legend': 'per-series', 'legend_swatch': 'line'}):
        fig = _ladder(dots=False)
        apply_figure_options(fig, {**options, 'names': NAMES})
        assert _legend(fig) == [(0, 'integration', 'lines'), (1, 'retention', 'lines'), (2, 'selection', 'lines')]


def test_title_and_axis_titles():
    fig = _ladder()
    apply_figure_options(fig, {'title': False, 'x_title': 'k̄', 'y_title': 'log₁₀ NMSE'})
    assert fig.layout.title.text is None
    assert (fig.layout.xaxis.title.text, fig.layout.yaxis.title.text) == ('k̄', 'log₁₀ NMSE')


def test_hlines_are_labelled_legend_lines_over_the_data_span():
    """A reference line is a trace in the legend (it scales as legend furniture, not as data text)."""
    fig = _ladder()
    apply_figure_options(fig, {'hlines': [{'y': 1.34, 'label': 'noise'}, {'y': 1.0}]})
    ref = [t for t in fig.data if t.mode == 'lines' and t.y is not None and len(set(t.y)) == 1 and t.y[0] in (1.34, 1.0)]
    assert [(list(t.x), t.y[0], t.name, t.showlegend) for t in ref] == [([1, 2], 1.34, 'noise', True), ([1, 2], 1.0, None, False)]


def test_legend_pos_puts_the_legend_inside_on_the_background():
    fig = _ladder()
    fig.update_layout(template='plotly_dark')
    apply_figure_options(fig, {'legend_pos': {'x': 0.99, 'y': 0.03, 'xanchor': 'right', 'yanchor': 'bottom'}})
    lg = fig.layout.legend
    assert (lg.x, lg.y, lg.xanchor, lg.yanchor) == (0.99, 0.03, 'right', 'bottom')
    assert lg.bgcolor == 'rgba(17, 17, 17, 0.8)'          # plotly_dark's paper colour, translucent
