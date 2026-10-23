import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from project.boolean_reservoir.code.reservoir import BatchedTensorHistoryWriter
from scipy.stats import zscore
from project.boolean_reservoir.code.utils.load_save import make_combo_column
from project.boolean_reservoir.code.utils.categorical_ordering import grayish_sort
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib
matplotlib.use('Agg')

def plot_train_history(path, history):
    history_df = pd.DataFrame(history)
    history_melted = history_df.melt(id_vars=['epoch'], value_vars=sorted([c for c in history_df.columns if c != 'epoch']), 
                                    var_name='metric', value_name='value')
    fig, ax1 = plt.subplots()

    loss_plot = sns.lineplot(data=history_melted[history_melted['metric'].str.contains('loss')], 
                             x='epoch', y='value', hue='metric', ax=ax1)
    loss_plot.legend(loc='upper left')

    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax2 = ax1.twinx()

    accuracy_plot = sns.lineplot(data=history_melted[history_melted['metric'].str.contains('accuracy')], 
                                 x='epoch', y='value', hue='metric', ax=ax2, linestyle='--')
    accuracy_plot.legend(loc='upper right')
    ax2.set_ylabel('Accuracy')
    fig.suptitle("Loss and Accuracy")
    fig.tight_layout()
    path = Path(path) / 'visualizations' 
    path.mkdir(parents=True, exist_ok=True)
    file = f"training.png"
    print('making train history plots:', path / file)
    plt.savefig(path / file, bbox_inches='tight')

def _preprocess(df, feature_cols):
    num_cols = df[feature_cols].select_dtypes(include='number').columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include='object').columns.tolist()
    transformers = [('num', StandardScaler(), num_cols)]
    if cat_cols:
        transformers.append(('cat', OneHotEncoder(sparse_output=False), cat_cols))
    preprocessor = ColumnTransformer(transformers)
    X = preprocessor.fit_transform(df[feature_cols])
    return X, preprocessor, num_cols, cat_cols


def plot_pca(df, feature_cols, target_col, out_path):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    df = df.reset_index(drop=True)
    X, preprocessor, num_cols, cat_cols = _preprocess(df, feature_cols)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    pc_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    pc_df[target_col] = df[target_col].values

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pc_df, x='PC1', y='PC2', hue=target_col, palette='viridis', s=100, alpha=0.7)
    plt.title('PCA of Parameters')
    plt.savefig(out_path / 'pca.png', bbox_inches='tight')
    plt.close()

    loadings = pca.components_.T
    feature_names = num_cols[:]
    if cat_cols:
        feature_names += preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    loading_df = pd.DataFrame(loadings, index=feature_names, columns=['PC1', 'PC2'])
    plt.figure(figsize=(max(4, len(feature_names) * 0.5), 6))
    sns.heatmap(loading_df, annot=True, cmap='coolwarm', cbar=True)
    plt.title('Parameter Contributions')
    plt.savefig(out_path / 'pca_legend.png', bbox_inches='tight')
    plt.close()


def plot_variance_explained(df, feature_cols, target_col, out_path):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    eta2 = {}
    for col in feature_cols:
        valid = df[[col, target_col]].dropna()
        if valid.empty or valid[col].nunique() < 2:
            continue
        y = valid[target_col]
        ss_total = ((y - y.mean()) ** 2).sum()
        if ss_total == 0:
            continue
        series = valid[col]
        if series.dtype.kind in 'fc' and series.nunique() > 10:
            series = pd.qcut(series, q=5, duplicates='drop').astype(str)
        groups = valid.assign(**{col: series}).groupby(col)[target_col]
        ss_between = sum(len(g) * (g.mean() - y.mean()) ** 2 for _, g in groups)
        eta2[col] = ss_between / ss_total

    if not eta2:
        print(f'plot_variance_explained: no variance data for {target_col}')
        return
    eta2_series = pd.Series(eta2).sort_values()
    plt.figure(figsize=(8, max(4, len(eta2_series) * 0.35)))
    eta2_series.plot.barh()
    plt.xlabel('η² (fraction of variance explained)')
    plt.title(f'Variance explained in {target_col}')
    plt.tight_layout()
    plt.savefig(out_path / 'variance_explained.png', bbox_inches='tight')
    plt.close()


def plot_metric_vs_parameter(df, feature_cols, target_col, out_path):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    for col in feature_cols:
        n_unique = df[col].nunique()
        if n_unique < 2:
            continue
        plt.figure(figsize=(8, 6))
        if n_unique > 10:
            sns.scatterplot(data=df, x=col, y=target_col)
        else:
            sns.boxplot(data=df, x=col, y=target_col)
        plt.title(f'{target_col} vs {col}')
        plt.tight_layout()
        plt.savefig(out_path / f'{col}.png', bbox_inches='tight')
        plt.close()


def plot_grid_search(
    df: pd.DataFrame,
    out_path: Path,
    target: str = 'T_accuracy',
    plots: list | None = None,
    label: str | None = None,
):
    out_path = Path(out_path) / (label or target)

    if 'T_loss' in df.columns:
        df = df[df['T_loss'] != float('inf')]
    df = df.loc[:, [df[c].nunique() > 1 for c in df.columns]]
    seed_cols = [c for c in df.columns if 'seed' in c.lower()]
    df = df.drop(columns=seed_cols, errors='ignore')

    if target not in df.columns or df[target].isna().all():
        print(f'plot_grid_search: target {target!r} has no data, aborting')
        return
    df = df.dropna(subset=[target])

    # Outcome metrics and metadata — everything else is a feature/hyperparameter
    outcome_cols = {'T_accuracy', 'T_loss', 'kqgr_kq', 'kqgr_gr', 'kqgr_delta'}
    excluded = outcome_cols | {c for c in df.columns if c.startswith(('L_', 'M_', 'i', 'j', 'params'))}
    feature_cols = [c for c in df.columns if c not in excluded and c != target]

    out_path.mkdir(parents=True, exist_ok=True)
    print('making grid search plots:', out_path)

    all_plots = {'pca', 'variance_explained', 'metric_vs_parameter'}
    to_run = set(plots or all_plots)

    if 'pca' in to_run:
        plot_pca(df, feature_cols, target, out_path / 'pca')
        plot_pca(df, feature_cols + [target], target, out_path / 'pca_w_target')
    if 'variance_explained' in to_run:
        plot_variance_explained(df, feature_cols, target, out_path)
    if 'metric_vs_parameter' in to_run:
        plot_metric_vs_parameter(df, feature_cols, target, out_path / 'metric_vs_parameter')

def plot_histogram_of_top_percentile_vs_config_id(path, df, top_percentile=0.1):
    threshold = df['T_accuracy'].quantile(1-top_percentile)
    config_ids = df[df['T_accuracy'] >= threshold]['config']
    config_counts = config_ids.value_counts().sort_index()
    top_config_id = config_counts.idxmax()
    print(f"Config ID with highest frequency: {top_config_id}, Count: {config_counts[top_config_id]}")
    
    plt.figure(figsize=(10, 6))
    plt.bar(config_counts.index, config_counts.values, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Config ID')
    plt.ylabel('Frequency')
    plt.title(f'Frequency of Config IDs in Top {int(top_percentile*100)}% of Accuracy (≥{threshold:.4f})')
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(rotation=45 if len(config_counts) > 10 else 0)
    plt.tight_layout()
    plt.savefig(path / 'histogram_accuracy_vs_config.png', bbox_inches='tight')
    plt.close()


# OUTDATED — not actively used
def plot_dynamics_history(path):
    path = Path(path)
    save_path = path / 'visualizations' 
    save_path.mkdir(parents=True, exist_ok=True)
    load_dict, history, expanded_meta, meta = BatchedTensorHistoryWriter(path / 'history').reload_history()
    # print(meta)
    # print('full history:', history.shape)
    expanded_meta = expanded_meta[expanded_meta['phase'].isin(['reservoir_layer', 'output_layer'])]
    history = history[expanded_meta.index].numpy()
    # print('filtered history:', history.shape)

    # normalize and perform dimension reduction
    history_normalized = zscore(history, axis=0)
    history_normalized = np.nan_to_num(history_normalized) # columns with all 0's or 1's will divide by zero as variance = 0
    n_components = 2

    # TODO transpose for space and time pca and break up into smaller parts
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(history_normalized)
    df = pd.DataFrame(embedding, columns=[f'PC{i+1}' for i in range(n_components)], index=expanded_meta.index)
    df = pd.concat([df, expanded_meta], axis=1)
    df['hue_tuple'] = df[['phase', 's', 'f']].apply(tuple, axis=1)

    # print("Explained variance by each component:")
    # print(embedding.explained_variance_ratio_)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='hue_tuple', palette='viridis', s=100, alpha=0.7)
    plt.legend(title='(phase, step, feature)')
    plt.title('PCA of states over time')
    file = f"pca.png"
    plt.savefig(save_path / file, bbox_inches='tight')

    tsne = TSNE(n_components=n_components, perplexity=30, learning_rate=200)
    embedding = tsne.fit_transform(history_normalized)
    df = pd.DataFrame(embedding, columns=[f'PC{i+1}' for i in range(n_components)], index=expanded_meta.index)
    df = pd.concat([df, expanded_meta], axis=1)
    df['hue_tuple'] = df[['phase', 's', 'f']].apply(tuple, axis=1)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='hue_tuple', palette='viridis', s=100, alpha=0.7)
    plt.legend(title='(phase, step, feature)')
    plt.title('tSNE of states over time')
    file = f"tsne.png"
    plt.savefig(save_path / file, bbox_inches='tight')

def plot_activity_trace(path, save_path=None, file_name="activity_trace_with_phase.png",
                        width=1200, height=800, fmt=None, active_phases=None):
    from dashboard.dashboards.trace import TracePlotDashboard
    from dashboard.dashboards.theme import register_templates
    register_templates()  # the dashboard's plotly themes; a running app registers them itself

    path = Path(path)
    save_path = Path(save_path) if save_path else path / 'visualizations'
    save_path.mkdir(parents=True, exist_ok=True)
    fmt = fmt or Path(file_name).suffix.lstrip('.') or 'png'

    dashboard = TracePlotDashboard(safe_roots=[str(path.parent), '/out', '/tmp'])
    img_bytes = dashboard.export_image(str(path), width, height, fmt, active_phases)
    (save_path / file_name).write_bytes(img_bytes)

# OUTDATED — not actively used
def plot_reconstructed_manifold(path, adjacency_matrix):
    path = Path(path)
    save_path = path / 'visualizations' 
    save_path.mkdir(parents=True, exist_ok=True)

    # Create a figure and a 3D Axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color map object from Seaborn
    cmap = sns.color_palette("viridis", as_cmap=True)

    # Plot the point cloud
    mds = MDS(n_components=3, dissimilarity="precomputed")
    reconstructed_points = mds.fit_transform(adjacency_matrix)
    x, y, z = reconstructed_points
    sc = ax.scatter(x, y, z, c=np.sqrt(x**2 + y**2 + z**2), cmap=cmap, s=20)

    # Add color bar which maps values to colors
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
    cbar.set_label('Color intensity')

    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Point Cloud Visualization with Seaborn colormap')
    file = f"reconstructed_manifold.png"
    plt.savefig(save_path / file, bbox_inches='tight')

def plot_predictions_and_labels(path, y_hat, y, tolerance=0.1, axis_limits=[0, 1], name=None,
                                 max_samples=None, color_tolerance=None, mode='spatial', fmt='png'):
    import numpy as np
    y_hat_np = np.asarray(y_hat)
    y_np = np.asarray(y)
    if max_samples is not None and len(y_np) > max_samples:
        idx = np.random.choice(len(y_np), max_samples, replace=False)
        y_hat_np = y_hat_np[idx]
        y_np = y_np[idx]
    num_dims = y_hat_np.shape[1]
    if num_dims == 1:
        sort_order = y_np[:, 0].argsort()
        y_hat_np = y_hat_np[sort_order]
        y_np = y_np[sort_order]

    # Per-sample RMSE (consistent with MSE training criterion). tolerance is defined against
    # raw Euclidean distance (see EuclideanDistanceAccuracy), so scale it down by sqrt(num_dims)
    # to land on the same saturation point.
    thresh = color_tolerance if color_tolerance is not None else tolerance
    rmse_thresh = thresh / np.sqrt(num_dims)
    errors = np.sqrt(np.mean((y_hat_np - y_np) ** 2, axis=1))
    color_val = np.clip(errors / rmse_thresh, 0, 1)
    cmap = 'RdYlGn_r'

    if mode == 'error':
        # Single-panel view: index sorted by target radius (distance from origin) vs RMSE.
        # Generalizes the 1D "sorted by value" view — radius is the natural 2D+ analog of value.
        radius = np.sqrt(np.sum(y_np ** 2, axis=1))
        order = radius.argsort()
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(range(len(order)), errors[order], c=color_val[order], cmap=cmap,
                         vmin=0, vmax=1, alpha=0.5, s=10)
        ax.grid(True)
        ax.set_title('Per-sample RMSE (sorted by target radius)')
        ax.set_xlabel('Index (sorted by ‖y‖)')
        ax.set_ylabel('RMSE')
        fig.colorbar(sc, ax=ax, label='RMSE / threshold (saturates at 1)')

        path = Path(path) / 'visualizations'
        path.mkdir(parents=True, exist_ok=True)
        file = f"{name}_error.{fmt}" if name else f"{num_dims}D_error.{fmt}"
        plt.savefig(path / file, bbox_inches='tight')
        plt.close(fig)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    y_index = range(len(y_hat_np))

    sc1 = ax1.scatter(
        y_hat_np[:, 0], y_hat_np[:, 1] if num_dims > 1 else y_index,
        c=color_val, cmap=cmap, vmin=0, vmax=1, alpha=0.3, s=10,
    )
    ax1.grid(True)
    ax1.set_title('Predicted coordinates')
    ax1.set_xlabel(r'$\hat{x}$')
    ax1.set_ylabel(r'$\hat{y}$' if num_dims > 1 else 'Index')
    ax1.set_xlim(axis_limits)
    if num_dims > 1:
        ax1.set_ylim(axis_limits)

    ax2.scatter(
        y_np[:, 0], y_np[:, 1] if num_dims > 1 else y_index,
        c=color_val, cmap=cmap, vmin=0, vmax=1, alpha=0.3, s=10,
    )
    ax2.grid(True)
    ax2.set_title('Target coordinates')
    ax2.set_xlabel(r'${x}$')
    ax2.set_ylabel(r'${y}$' if num_dims > 1 else 'Index')
    ax2.set_xlim(axis_limits)
    if num_dims > 1:
        ax2.set_ylim(axis_limits)

    fig.colorbar(sc1, ax=[ax1, ax2], label='RMSE / threshold (saturates at 1)')

    path = Path(path) / 'visualizations'
    path.mkdir(parents=True, exist_ok=True)
    file = f"{name}_predictions_versus_labels.{fmt}" if name else f"{num_dims}D_predictions_versus_labels.{fmt}"
    plt.savefig(path / file, bbox_inches='tight')
    plt.close(fig)



