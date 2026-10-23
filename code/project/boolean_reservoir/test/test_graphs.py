import pytest
import numpy as np
from project.boolean_reservoir.code.graph import generate_adjacency_matrix


@pytest.mark.parametrize(
    "n_nodes, k_min, k_avg, k_max, self_loops",
    [
        (1000, 2, 3, 4, 0),
        (1000, 3, 3, 4, 0),
        (1000, 2, 3, 3, 0),
        (1000, 3, 3, 3, 0),
        (1000, 2, 3, 4, 1),
        (1000, 3, 3, 4, 1),
        (1000, 2, 3, 3, 1),
        (1000, 3, 3, 3, 1),
        (1000, 1, 2, 3, 0),
        (1000, 2, 3, 4, 0.1),
        (1000, 1, 2, 3, 0.1),
        (1000, 2, 3, 4, 0.01),
        (1000, 1, 2, 3, 0.01),
        (1000, 2, 3, 4, 0.9),
        (1000, 1, 2, 3, 0.9),
        (1000, 1, 2, 3, 1),
        (1000, 1, 500, 1000, 0.1),
        (1000, 1, 1000, 1000, 1),
        (1000, 0, 100, 500, 0),
        (1000, 0, 100, 500, 1),
        (500, 0, 1, 10, 0.95),
    ]
)
def test_generate_adjacency_matrix(n_nodes, k_min, k_avg, k_max, self_loops):
    adj_matrix = generate_adjacency_matrix(n_nodes, k_min, k_avg, k_max, self_loops)

    assert adj_matrix.shape == (n_nodes, n_nodes)
    assert adj_matrix.sum() == round(n_nodes * k_avg), "total edges mismatch"
    assert np.diagonal(adj_matrix).sum() == round(n_nodes * self_loops), "self-loop count mismatch"
    assert (adj_matrix.sum(axis=0) <= k_max).all(), "k_max violated"
    assert (adj_matrix.sum(axis=0) >= k_min).all(), "k_min violated"

    in_degrees = adj_matrix.sum(axis=0)
    np.testing.assert_allclose(
        in_degrees.mean(), k_avg, atol=0.5,
        err_msg=f"mean in-degree {in_degrees.mean():.2f} too far from k_avg={k_avg}"
    )


def _plot_in_degree_density_grid(n_samples, out_path):
    import matplotlib.pyplot as plt
    from labellines import labelLines

    k_avg_values = range(1, 9)
    k_max = 25
    x_lim = 15
    self_loops = 0
    x = np.arange(0, k_max + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, n_nodes in enumerate([32, 512]):
        for col, mode in enumerate(['homogeneous', 'heterogeneous']):
            ax = axes[row, col]
            for k_avg in k_avg_values:
                k_min_i, k_max_i = (k_avg, k_avg) if mode == 'homogeneous' else (0, k_max)
                counts = np.zeros(k_max + 1)
                for _ in range(n_samples):
                    adj = generate_adjacency_matrix(n_nodes, k_min_i, k_avg, k_max_i, self_loops)
                    counts += np.bincount(adj.sum(axis=0), minlength=k_max + 1)
                density = counts / counts.sum()
                ax.plot(x, density, label=str(k_avg))
            ax.set_xlim(0, x_lim)
            ax.set_ylim(0, 1)
            ax.set_xlabel('in_degree')
            ax.set_ylabel('density')
            ax.set_title(f'{mode}, n_nodes={n_nodes}, k_max={k_max}')
            if col == 0:  # homogeneous: label on ascending edge at y=0.5
                xvals = []
                for line in ax.get_lines():
                    xd, yd = line.get_xdata(), line.get_ydata()
                    above = np.where(yd >= 0.5)[0]
                    if len(above) == 0 or above[0] == 0:
                        xvals.append(xd[0])
                    else:
                        i = above[0]
                        t = (0.5 - yd[i - 1]) / (yd[i] - yd[i - 1])
                        xvals.append(xd[i - 1] + t * (xd[i] - xd[i - 1]))
            else:  # heterogeneous: label at each line's peak
                xvals = list(k_avg_values)
            labelLines(ax.get_lines(), xvals=xvals, zorder=2.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    from pathlib import Path

    out_dir = Path(__file__).parents[3] / 'out' / 'test' / 'boolean_reservoir' / 'graph'
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_in_degree_density_grid(n_samples=50, out_path=out_dir / 'in_degree_distributions_averaged.svg')
    _plot_in_degree_density_grid(n_samples=1,  out_path=out_dir / 'in_degree_distributions_single_sample.svg')
    print("Plots written to", out_dir)
