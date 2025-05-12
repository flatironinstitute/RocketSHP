# %% Defines
import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from rocketshp.metrics import graph_diffusion_distance, ipsen_mikhailov_distance


def display_network(
    A: np.ndarray,
    title: str = "Network",
    node_color: str = "lightblue",
    edge_color: str = "gray",
    seed: int = 42,
    ax=None,
):
    """
    Plot a network represented by an adjacency matrix.

    Parameters:
    -----------
    A : numpy.ndarray
        Weighted adjacency matrix
    title : str, optional
        Title of the plot (default="Network")
    node_color : str, optional
        Color of the nodes (default='lightblue')
    edge_color : str, optional
        Color of the edges (default='gray')
    """
    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, seed=seed)  # positions for all nodes
    nx.draw(
        G, pos, with_labels=True, node_color=node_color, edge_color=edge_color, ax=ax
    )
    if ax is not None:
        ax.set_title(title)
    else:
        plt.title(title)
        plt.show()


def pairwise_correlation_to_network(
    A: np.ndarray, thresh: float = 0.5, title: str = "Network", seed: int = 42, ax=None
):
    """
    Convert a pairwise correlation matrix to a network and plot it.

    """
    sparse = A.copy()
    sparse[A < thresh] = 0
    np.fill_diagonal(sparse, 0)

    # node color by residue index
    node_color = np.arange(sparse.shape[0])
    node_color = node_color / node_color.max()
    node_color = plt.cm.RdYlBu(node_color)
    node_color = node_color[:, :3]  # Keep only RGB channels
    node_color = node_color.tolist()

    display_network(
        sparse, title=title, node_color=node_color, edge_color="gray", seed=seed, ax=ax
    )


def build_allosteric_network(
    gcc_lmi: np.ndarray, ca_dist: np.ndarray, distance_cutoff: float = 8.0
):
    dist_thresh_nm = distance_cutoff / 10.0

    mask = ca_dist < dist_thresh_nm
    masked_net = gcc_lmi * mask
    np.fill_diagonal(masked_net, 0)  # remove self-edges
    G = nx.from_numpy_array(masked_net)

    return G


def cluster_network(G: nx.Graph, k: int = 5):
    """
    Cluster the network using Girvan-Newman algorithm.
    """
    comp = nx.community.girvan_newman(G)
    limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    for communities in limited:
        clusts = tuple(sorted(c) for c in communities)
    return clusts


def calculate_centrality(G: nx.Graph):
    """
    Calculate centrality measures for the network.
    """
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    degree = nx.degree_centrality(G)

    return betweenness, closeness, degree


def plot_network_clusters(
    G: nx.Graph,
    clusters: list,
    title: str = "Network Clusters",
    output_path: str = "network_clusters.png",
):
    """
    Plot the network with clusters highlighted.
    """
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(10, 8))

    node_color = []
    for i, cluster in enumerate(clusters):
        node_color.extend([i] * len(cluster))
        # nx.draw_networkx_nodes(G, pos, nodelist=cluster, node_color=node_color, label=f"Cluster {i+1}")

    # display_network(sparse, title=title, node_color=node_color, edge_color="gray", seed=seed, ax=ax)

    nx.draw(G, pos, with_labels=True, node_color=node_color, edge_color="gray", ax=ax)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


# Example usage:
# %% Create two sample weighted matrices
if __name__ == "__main__":
    A = np.array([[0, 0.5, 0.2, 0], [0.5, 0, 0.7, 1], [0.2, 0.7, 0, 0], [0, 1, 0, 0]])
    B = np.array(
        [[0, 0.4, 0.3, 0], [0.4, 0, 0.5, 0], [0.3, 0.5, 0, 3.4], [0, 0, 3.4, 0]]
    )

    # Plot each network
    display_network(A, title="Network A")
    display_network(B, title="Network B")

    # Compute the distances
    gdd = graph_diffusion_distance(A, B)
    im_dist = ipsen_mikhailov_distance(A, B)

    print(f"Graph Diffusion Distance: {gdd:.6f}")
    print(f"Ipsen-Mikhailov Distance: {im_dist:.6f}")

# %%
