import os, torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch.nn as nn

from torch.utils.data import DataLoader

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModel
from PIL import Image

import numpy as np
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

import leidenalg
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.plots import plot_convergence, plot_objective

from sklearn.decomposition import PCA
import umap

import colorcet as cc
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import Counter

"""
Utilities for image embedding, clustering, and visualization.

The module provides helpers to:
- load image files into PyTorch datasets,
- extract normalized transformer embeddings,
- build a k-nearest-neighbor graph over those embeddings,
- cluster the graph with Leiden community detection, and
- visualize optimization and low-dimensional projections.
"""

class FileListDataset(Dataset):
    """
    Dataset that lazily loads RGB images from a list of file paths.

    Parameters
    ----------
    file_list : Sequence[str | os.PathLike]
        Paths to images that should be opened on demand.
    transform : callable, optional
        Optional transform applied to each PIL image before returning it.
    """
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

def collate_fn(batch):
    """
    Collate `(image, path)` samples into parallel image and path lists.

    Parameters
    ----------
    batch : Sequence[tuple]
        Iterable of `(image, path)` pairs produced by `FileListDataset`.

    Returns
    -------
    tuple[list, list]
        A pair containing the images and the corresponding file paths.
    """
    imgs, paths = zip(*batch)
    return list(imgs), list(paths)

def extract_features(image_files, 
                     model_name = "microsoft/swin-tiny-patch4-window7-224"): # "openai/clip-vit-base-patch16"): 
    """
    Extract normalized feature embeddings for a list of images.

    Parameters
    ----------
    image_files : Sequence[str | os.PathLike]
        Image paths to embed.
    model_name : str, optional
        Hugging Face model identifier used for the processor and backbone.
        The model is expected to expose a `pooler_output` tensor.

    Returns
    -------
    tuple[list[str | os.PathLike], numpy.ndarray]
        The processed paths in dataloader order and a 2D array of
        L2-normalized embeddings with one row per image.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    dataset = FileListDataset(image_files)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    features, paths = [], []

    with torch.no_grad():
        for imgs, img_paths in dataloader:
            inputs = processor(images=imgs, return_tensors="pt").to(device)

            outputs = model(**inputs)
            embeddings = outputs.pooler_output
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            features.append(embeddings.cpu())
            paths.extend(img_paths)

    features = torch.cat(features).numpy()

    return paths, features

def build_knn_graph(features, n_neighbors=5):
    """
    Build an undirected k-nearest-neighbor graph from feature vectors.

    Parameters
    ----------
    features : array-like of shape (n_samples, n_features)
        Feature matrix used to compute nearest neighbors.
    n_neighbors : int, optional
        Number of neighbors to connect for each sample before the graph is
        symmetrized.

    Returns
    -------
    igraph.Graph
        Undirected graph with one vertex per feature vector.
    """
    # build KNN graph
    A = kneighbors_graph(
        features,
        n_neighbors=n_neighbors,
        mode='connectivity',
        include_self=False
    )
    A = A.maximum(A.T)  # symmetrise

    # build igraph
    sources, targets = A.nonzero()
    g = ig.Graph(
        n=A.shape[0],
        edges=np.column_stack([sources, targets]),
        directed=False
    )
    return g

def leiden_with_resolution(g, resolution, seed=42):
    """
    Run Leiden community detection for a fixed resolution parameter.

    Parameters
    ----------
    g : igraph.Graph
        Graph to partition.
    resolution : float
        Resolution parameter passed to
        `leidenalg.RBConfigurationVertexPartition`.
    seed : int, optional
        Random seed used by Leiden.

    Returns
    -------
    tuple[float, int, leidenalg.VertexPartition]
        Modularity score, number of discovered clusters, and the fitted
        partition object.
    """
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=seed
    )
    modularity = partition.modularity
    n_clusters = len(np.unique(partition.membership))
    return modularity, n_clusters, partition

def find_best_resolution(
    features,
    resolutions=np.linspace(0.1, 2.0, 20),
    n_neighbors=5,
    repeats=1,
    seed=42
):
    """
    Search a fixed resolution grid for the highest-modularity partition.

    Parameters
    ----------
    features : array-like of shape (n_samples, n_features)
        Feature matrix to cluster.
    resolutions : iterable of float, optional
        Resolution values to evaluate.
    n_neighbors : int, optional
        Neighborhood size used to build the graph once before the sweep.
    repeats : int, optional
        Number of Leiden runs per resolution. Additional runs increment the
        provided seed by `rep`.
    seed : int, optional
        Base random seed for Leiden.

    Returns
    -------
    dict
        Dictionary containing the best resolution, its modularity, the best
        partition, and a list of `(resolution, mean_modularity, n_clusters)`
        tuples for every evaluation.
    """
    g = build_knn_graph(features, n_neighbors=n_neighbors)

    results = []
    best_partition = None
    best_modularity = -np.inf
    best_resolution = None

    print("🔍 Suche optimalen Resolution-Parameter...\n")

    for r in tqdm(resolutions):
        modularity_scores = []

        for rep in range(repeats):
            modularity, n_clusters, partition = leiden_with_resolution(g, r, seed + rep)
            modularity_scores.append(modularity)

        mean_modularity = np.mean(modularity_scores)
        results.append((r, mean_modularity, n_clusters))

        if mean_modularity > best_modularity:
            best_modularity = mean_modularity
            best_resolution = r
            best_partition = partition

    return {
        "best_resolution": best_resolution,
        "best_modularity": best_modularity,
        "best_partition": best_partition,
        "results": results
    }

def find_best_parameters_bayesian(
    features,
    n_neighbors_range=(3, 30),
    resolution_range=(0.1, 2.0),
    n_calls=50,
    n_random_starts=10,
    repeats=3,
    seed=42,
    verbose=True
):
    """
    Optimize `n_neighbors` and Leiden `resolution` with Bayesian search.

    Parameters
    ----------
    features : array-like of shape (n_samples, n_features)
        Feature matrix to cluster.
    n_neighbors_range : tuple[int, int], optional
        Inclusive search bounds for `n_neighbors`.
    resolution_range : tuple[float, float], optional
        Inclusive search bounds for the Leiden resolution parameter.
    n_calls : int, optional
        Total number of objective evaluations.
    n_random_starts : int, optional
        Number of random evaluations performed before model-guided search.
    repeats : int, optional
        Number of repeated clustering runs per parameter configuration.
    seed : int, optional
        Random seed for the optimizer and Leiden runs.
    verbose : bool, optional
        Whether to print progress and per-evaluation metrics.

    Returns
    -------
    dict
        Dictionary containing the best parameters, best partition, the raw
        `skopt` optimization result, and an evaluation history.
    """
    
    # history for all evaluations
    history = []
    
    def objective(params):
        """
        Objective function: Will be minimized by Bayesian Optimization
        """
        n_neighbors = int(params[0])
        resolution = params[1]
        
        modularity_scores = []
        
        for rep in range(repeats):
            try:
                # build graph
                g = build_knn_graph(features, n_neighbors=n_neighbors)
                
                # Clustering
                modularity, n_clusters, partition = leiden_with_resolution(
                    g, resolution, seed + rep
                )
                
                modularity_scores.append(modularity)
                
            except Exception as e:
                if verbose:
                    print(f"⚠️ Error at n={n_neighbors}, r={resolution:.3f}: {e}")
                return 1.0  # Bad score on error
        
        mean_modularity = np.mean(modularity_scores)
        std_modularity = np.std(modularity_scores)
        
        # save to history
        history.append({
            'n_neighbors': n_neighbors,
            'resolution': resolution,
            'mean_modularity': mean_modularity,
            'std_modularity': std_modularity,
            'n_clusters': n_clusters
        })
        
        if verbose:
            print(f"n={n_neighbors:2d}, r={resolution:.3f} → "
                  f"Modularity={mean_modularity:.4f}±{std_modularity:.4f}, "
                  f"Clusters={n_clusters}")
        
        # minimize = return negative modularity
        return -mean_modularity
    
    # define search space
    space = [
        Integer(n_neighbors_range[0], n_neighbors_range[1], name='n_neighbors'),
        Real(resolution_range[0], resolution_range[1], name='resolution')
    ]
    
    print("Starting Bayesian Optimization...")
    print(f" Search space: n_neighbors={n_neighbors_range}, resolution={resolution_range}")
    print(f" Evaluations: {n_calls} (including {n_random_starts} random)\n")
    
    # run Bayesian optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        random_state=seed,
        verbose=False,
        acq_func='EI',  # Expected Improvement
        n_jobs=1  # Parallelization if desired
    )
    
    # extract best parameters
    best_n_neighbors = int(result.x[0])
    best_resolution = result.x[1]
    best_modularity = -result.fun  # Make positive again
    
    # recompute best partition
    g = build_knn_graph(features, n_neighbors=best_n_neighbors)
    _, _, best_partition = leiden_with_resolution(g, best_resolution, seed)
    
    return {
        'best_n_neighbors': best_n_neighbors,
        'best_resolution': best_resolution,
        'best_modularity': best_modularity,
        'best_partition': best_partition,
        'optimization_result': result,
        'history': history
    }

def plot_optimization_results(result, output_dir=None):
    """
    Visualize Bayesian search progress and save plots if requested.

    Parameters
    ----------
    result : dict
        Output produced by `find_best_parameters_bayesian`.
    output_dir : str | os.PathLike, optional
        Directory where PNG versions of the plots should be written.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Convergence Plot
    from skopt.plots import plot_convergence
    plot_convergence(result['optimization_result'], ax=axes[0, 0])
    axes[0, 0].set_title('Convergence Plot')
    
    # 2. Objective Plot (2D)
    from skopt.plots import plot_objective
    plot_objective(result['optimization_result'], dimensions=['n_neighbors', 'resolution'])
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'objective_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. History: Modularity over time
    history = result['history']
    modularities = [h['mean_modularity'] for h in history]
    axes[0, 1].plot(modularities, 'o-', alpha=0.6)
    axes[0, 1].axhline(result['best_modularity'], color='red', 
                       linestyle='--', label='Best')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Modularity')
    axes[0, 1].set_title('Modularity over Iterations')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 4. Parameter Space Exploration
    n_neighbors = [h['n_neighbors'] for h in history]
    resolutions = [h['resolution'] for h in history]
    modularities = [h['mean_modularity'] for h in history]
    
    scatter = axes[1, 0].scatter(n_neighbors, resolutions, 
                                  c=modularities, s=100, 
                                  cmap='viridis', alpha=0.6)
    axes[1, 0].scatter(result['best_n_neighbors'], 
                       result['best_resolution'],
                       s=300, marker='*', color='red', 
                       edgecolors='black', linewidths=2,
                       label='Best', zorder=5)
    axes[1, 0].set_xlabel('n_neighbors')
    axes[1, 0].set_ylabel('resolution')
    axes[1, 0].set_title('Parameter Space Exploration')
    axes[1, 0].legend()
    plt.colorbar(scatter, ax=axes[1, 0], label='Modularity')
    
    # 5. Number of clusters
    n_clusters = [h['n_clusters'] for h in history]
    axes[1, 1].plot(n_clusters, 'o-', alpha=0.6, color='green')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Number of Clusters')
    axes[1, 1].set_title('Cluster Count over Iterations')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'bayesian_optimization_results.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    if output_dir:
        print(f"\nAdditional plots saved in {output_dir}:")
        print("   - objective_plot.png")
        print("   - bayesian_optimization_results.png")

def analyze_sensitivity(features, result, n_samples=20, seed=42, output_dir=None):
    """
    Measure how modularity changes around the best Bayesian parameters.

    Parameters
    ----------
    features : array-like of shape (n_samples, n_features)
        Feature matrix to cluster.
    result : dict
        Output produced by `find_best_parameters_bayesian`.
    n_samples : int, optional
        Number of samples to evaluate for each one-dimensional sweep.
    seed : int, optional
        Random seed used for Leiden runs.
    output_dir : str | os.PathLike, optional
        Directory where the sensitivity plot should be written.

    Returns
    -------
    list[dict]
        Evaluation records for both `n_neighbors` and `resolution` sweeps.
    """
    best_n = result['best_n_neighbors']
    best_r = result['best_resolution']
    
    # Small variations around best parameters
    n_range = np.linspace(max(3, best_n - 5), best_n + 5, n_samples, dtype=int)
    r_range = np.linspace(max(0.1, best_r - 0.3), best_r + 0.3, n_samples)
    
    results = []
    
    print("\nSensitivity analysis...")
    
    # variation of n_neighbors
    for n in n_range:
        g = build_knn_graph(features, n_neighbors=int(n))
        mod, n_clust, _ = leiden_with_resolution(g, best_r, seed)
        results.append({
            'type': 'n_neighbors',
            'value': n,
            'modularity': mod,
            'n_clusters': n_clust
        })
    
    # variation of resolution
    g = build_knn_graph(features, n_neighbors=best_n)
    for r in r_range:
        mod, n_clust, _ = leiden_with_resolution(g, r, seed)
        results.append({
            'type': 'resolution',
            'value': r,
            'modularity': mod,
            'n_clusters': n_clust
        })
    
    # visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_data = [r for r in results if r['type'] == 'n_neighbors']
    r_data = [r for r in results if r['type'] == 'resolution']
    
    axes[0].plot([d['value'] for d in n_data], 
                 [d['modularity'] for d in n_data], 'o-')
    axes[0].axvline(best_n, color='red', linestyle='--', label='Optimal')
    axes[0].set_xlabel('n_neighbors')
    axes[0].set_ylabel('Modularity')
    axes[0].set_title('Sensitivity: n_neighbors')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot([d['value'] for d in r_data], 
                 [d['modularity'] for d in r_data], 'o-')
    axes[1].axvline(best_r, color='red', linestyle='--', label='Optimal')
    axes[1].set_xlabel('resolution')
    axes[1].set_ylabel('Modularity')
    axes[1].set_title('Sensitivity: resolution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'sensitivity_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

def compute_and_plot_pca(labels, features, output_dir=None):
    """
    Project features with PCA and visualize them colored by cluster label.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Cluster label for each feature vector.
    features : array-like of shape (n_samples, n_features)
        Feature matrix to project into two dimensions.
    output_dir : str | os.PathLike, optional
        Directory where `leiden_clustering_pca.png` should be saved.
    """
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, s=10)
    plt.title(f"Leiden Clustering (PCA, JPEG Features)")

    if output_dir:
        filename = os.path.join(output_dir, "leiden_clustering_pca.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    
    plt.show()
    plt.close()

def plot_embedding(labels, features, method="umap", annotate_clusters=False, output_file=None,):
    """
    Plot a 2D embedding of clustered features.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Cluster labels used for coloring points.
    features : array-like of shape (n_samples, n_features)
        Feature matrix to reduce into two dimensions.
    method : {"umap", "pca"}, optional
        Dimensionality-reduction method used before plotting.
    annotate_clusters : bool, optional
        If `True`, annotate representative points for a subset of clusters.
    output_file : str | os.PathLike, optional
        Path where the rendered plot should be written.
    """

    labels = np.asarray(labels)

    if method.lower() == "umap":
        reducer = umap.UMAP(
            n_components=2,
            random_state=42
        )
        reduced = reducer.fit_transform(features)
        x_label = "UMAP-1"
        y_label = "UMAP-2"
        title = "Leiden Clustering (UMAP)"
    elif method.lower() == "pca":
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(features)
        x_label = "PC1"
        y_label = "PC2"
        title = "Leiden Clustering (PCA)"
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Supported methods: 'umap', 'pca'."
        )

    unique_labels = np.sort(np.unique(labels))
    n_clusters = len(unique_labels)

    cmap = ListedColormap(cc.glasbey[:n_clusters])
    bounds = np.arange(n_clusters + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(reduced[:, 0], reduced[:, 1], 
                    c=labels, s=10, cmap=cmap, norm=norm, alpha=0.7)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Cluster")

    if n_clusters > 25:
        step = int(np.ceil(n_clusters / 25))
        shown_labels = unique_labels[::step]
    else:
        shown_labels = unique_labels

    cbar.set_ticks(shown_labels)
    cbar.set_ticklabels(shown_labels)

    if annotate_clusters:
        for cluster_id in shown_labels:
            mask = labels == cluster_id

            pts = reduced[mask]
            center = pts.mean(axis=0)

            # Punkt innerhalb des Clusters nahe am Schwerpunkt
            idx = np.argmin(
                np.sum((pts - center) ** 2, axis=1)
            )

            x_text, y_text = pts[idx]

            ax.text(
                x_text,
                y_text,
                str(cluster_id),
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                    pad=1,
                ),
            )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.tight_layout()

    if output_file:
        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    return fig
