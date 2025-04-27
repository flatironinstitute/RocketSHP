# %% Defines
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from rocketshp.metrics import graph_diffusion_distance, ipsen_mikhailov_distance

def display_network(A: np.ndarray, title: str = "Network", node_color: str = 'lightblue', edge_color: str = 'gray', seed: int = 42, ax=None):
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
    nx.draw(G, pos, with_labels=True, node_color=node_color, edge_color=edge_color, ax=ax)
    if ax is not None:
        ax.set_title(title)
    else:
        plt.title(title)
        plt.show()

def pairwise_correlation_to_network(A: np.ndarray, thresh: float = 0.5, title: str = "Network", seed: int = 42, ax=None):
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

    display_network(sparse, title=title, node_color=node_color, edge_color="gray", seed=seed, ax=ax)


# Example usage:
# %% Create two sample weighted matrices
if __name__ == "__main__":
    A = np.array([[0, 0.5, 0.2, 0], [0.5, 0, 0.7, 1], [0.2, 0.7, 0, 0], [0, 1, 0, 0]])
    B = np.array([[0, 0.4, 0.3, 0], [0.4, 0, 0.5, 0], [0.3, 0.5, 0, 3.4], [0, 0, 3.4, 0]])

    # Plot each network
    display_network(A, title="Network A")
    display_network(B, title="Network B")

    # Compute the distances
    gdd = graph_diffusion_distance(A, B)
    im_dist = ipsen_mikhailov_distance(A, B)

    print(f"Graph Diffusion Distance: {gdd:.6f}")
    print(f"Ipsen-Mikhailov Distance: {im_dist:.6f}")

# %%
