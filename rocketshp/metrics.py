import numpy as np
from scipy import linalg
from scipy.integrate import quad
from scipy.stats import pearsonr, spearmanr, wasserstein_distance_nd
from torch.nn.functional import kl_div, log_softmax


def graph_diffusion_distance(A, B, beta=1.0):
    """
    Compute the Graph Diffusion Distance between two weighted networks.

    Parameters:
    -----------
    A : numpy.ndarray
        First weighted adjacency matrix
    B : numpy.ndarray
        Second weighted adjacency matrix
    beta : float, optional
        Diffusion parameter controlling the time scale (default=1.0)

    Returns:
    --------
    float
        The Graph Diffusion Distance between networks A and B
    """
    # Ensure matrices are of the same shape
    if A.shape != B.shape:
        raise ValueError("Input matrices must have the same dimensions")

    # Compute Laplacian matrices
    D_A = np.diag(np.sum(A, axis=1))
    D_B = np.diag(np.sum(B, axis=1))
    L_A = D_A - A
    L_B = D_B - B

    # Compute matrix exponentials (heat kernels)
    H_A = linalg.expm(-beta * L_A)
    H_B = linalg.expm(-beta * L_B)

    # Compute Frobenius norm of the difference
    diff = H_A - H_B
    gdd = np.sqrt(np.sum(diff * diff))

    return gdd


def ipsen_mikhailov_distance(A, B, gamma=0.08):
    """
    Compute the Ipsen-Mikhailov distance between two weighted networks.

    Parameters:
    -----------
    A : numpy.ndarray
        First weighted adjacency matrix
    B : numpy.ndarray
        Second weighted adjacency matrix
    gamma : float, optional
        Scale parameter that determines the width of the Lorentzian distribution (default=0.08)

    Returns:
    --------
    float
        The Ipsen-Mikhailov distance between networks A and B
    """
    # Ensure matrices are of the same shape
    if A.shape != B.shape:
        raise ValueError("Input matrices must have the same dimensions")

    # Compute Laplacian matrices
    n = A.shape[0]
    D_A = np.diag(np.sum(A, axis=1))
    D_B = np.diag(np.sum(B, axis=1))
    L_A = D_A - A
    L_B = D_B - B

    # Compute eigenvalues (excluding the zero eigenvalue)
    eigvals_A = np.real(linalg.eigvals(L_A))
    eigvals_A = eigvals_A[eigvals_A > 1e-10]  # Remove zero eigenvalues

    eigvals_B = np.real(linalg.eigvals(L_B))
    eigvals_B = eigvals_B[eigvals_B > 1e-10]  # Remove zero eigenvalues

    # Compute spectral densities using Lorentzian distribution
    def spectral_density(omega, eigenvalues):
        K = 1.0 / (n - 1.0)  # Normalization constant
        density = 0.0
        for lambda_i in eigenvalues:
            density += K * gamma / ((omega - np.sqrt(lambda_i)) ** 2 + gamma**2)
        return density

    # Define the range of frequencies to integrate over
    # Typically from 0 to a value larger than the maximum eigenvalue
    max_eigval = max(np.max(eigvals_A), np.max(eigvals_B))
    omega_max = 2 * np.sqrt(max_eigval)

    # Compute the squared difference between spectral densities
    def integrand(omega):
        rho_A = spectral_density(omega, eigvals_A)
        rho_B = spectral_density(omega, eigvals_B)
        return (rho_A - rho_B) ** 2

    # Numerical integration
    result, _ = quad(integrand, 0, omega_max)

    # Return square root of the integral
    return np.sqrt(result)


def pearson(A, B):
    return pearsonr(A.flatten(), B.flatten())


def spearman(A, B):
    return spearmanr(A.flatten(), B.flatten())


def mse(A, B):
    return ((A.flatten() - B.flatten()) ** 2).mean()


def mae(A, B):
    return np.abs(A.flatten() - B.flatten()).mean()


def rmse(A, B):
    return np.sqrt(mse(A, B))


def wasserstein_2d(A, B):
    """
    Compute the Wasserstein distance between two distributions where each column is an observation

    Parameters:
    -----------
    A : numpy.ndarray
        First distribution (e.g., softmax probabilities)
    B : numpy.ndarray
        Second distribution (e.g., softmax probabilities)
    Returns:
    --------
    float
        The Wasserstein distance between distributions A and B
    """
    # Ensure matrices are of the same shape
    if A.shape != B.shape:
        raise ValueError("Input matrices must have the same dimensions")

    # Compute Wasserstein distance
    w_dist = wasserstein_distance_nd(A.T, B.T)

    return w_dist


def kl_divergence_2d(A, B):
    """
    Compute the Kullback-Leibler divergence between two distributions.

    Parameters:
    -----------
    A : torch.Tensor
        First distribution (e.g., softmax probabilities)
    B : torch.Tensor
        Second distribution (e.g., softmax probabilities)

    Returns:
    --------
    float
        The Kullback-Leibler divergence between distributions A and B
    """
    # Ensure matrices are of the same shape
    if A.shape != B.shape:
        raise ValueError("Input matrices must have the same dimensions")

    # Convert to log probabilities
    log_B = log_softmax(B, dim=1)

    # Compute KL divergence
    kl = kl_div(log_B, A, log_target=False, reduction="none")
    kl = kl.sum(dim=1).mean()

    return kl.item()
