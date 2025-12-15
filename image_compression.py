import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import svd

def load_image(image_path, grayscale=True):
    """
    Load image and convert to matrix
    
    Parameters:
    -----------
    image_path : str or file object
        Path to image or uploaded file
    grayscale : bool
        Convert to grayscale if True
    
    Returns:
    --------
    numpy.ndarray : Image matrix
    """
    try:
        img = Image.open(image_path)
        if grayscale:
            img = img.convert('L')  # Convert to grayscale
        return np.array(img, dtype=float)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def compress_image_svd(image_matrix, k):
    """
    Compress image using SVD with rank-k approximation
    
    Parameters:
    -----------
    image_matrix : numpy.ndarray
        Original image matrix
    k : int
        Number of singular values to keep (rank)
    
    Returns:
    --------
    numpy.ndarray : Compressed image matrix
    """
    # Perform SVD: A = U * Sigma * V^T
    U, sigma, Vt = svd(image_matrix, full_matrices=False)
    
    # Keep only top-k singular values
    U_k = U[:, :k]
    sigma_k = sigma[:k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct image: A_k = U_k * Sigma_k * Vt_k
    compressed = U_k @ np.diag(sigma_k) @ Vt_k
    
    # Clip values to valid range [0, 255]
    compressed = np.clip(compressed, 0, 255)
    
    return compressed

def calculate_compression_ratio(original_shape, k):
    """
    Calculate compression ratio
    
    Parameters:
    -----------
    original_shape : tuple
        (m, n) dimensions of original image
    k : int
        Rank used for compression
    
    Returns:
    --------
    float : Compression ratio as percentage
    """
    m, n = original_shape
    original_size = m * n
    compressed_size = k * (m + n + 1)
    
    ratio = (1 - compressed_size / original_size) * 100
    return ratio

def compute_quality_metrics(original, compressed):
    """
    Compute quality metrics (MSE and PSNR)
    
    Parameters:
    -----------
    original : numpy.ndarray
        Original image matrix
    compressed : numpy.ndarray
        Compressed image matrix
    
    Returns:
    --------
    dict : Dictionary with MSE and PSNR values
    """
    mse = np.mean((original - compressed) ** 2)
    
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return {'MSE': mse, 'PSNR': psnr}

def get_singular_values(image_matrix):
    """
    Get singular values of image matrix
    
    Parameters:
    -----------
    image_matrix : numpy.ndarray
        Image matrix
    
    Returns:
    --------
    numpy.ndarray : Singular values
    """
    _, sigma, _ = svd(image_matrix, full_matrices=False)
    return sigma

def plot_singular_values(sigma, k=None):
    """
    Plot singular values spectrum
    
    Parameters:
    -----------
    sigma : numpy.ndarray
        Singular values
    k : int, optional
        Highlight first k values
    
    Returns:
    --------
    matplotlib.figure.Figure : The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(sigma, 'b-', linewidth=2, label='Singular Values')
    
    if k is not None and k <= len(sigma):
        ax.axvline(x=k, color='r', linestyle='--', 
                   linewidth=2, label=f'k = {k}')
        ax.plot(range(k), sigma[:k], 'ro', markersize=5)
    
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Singular Value', fontsize=12)
    ax.set_title('Singular Value Spectrum', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_energy_retention(sigma):
    """
    Plot cumulative energy retention
    
    Parameters:
    -----------
    sigma : numpy.ndarray
        Singular values
    
    Returns:
    --------
    matplotlib.figure.Figure : The figure object
    """
    # Calculate cumulative energy
    energy = sigma ** 2
    cumulative_energy = np.cumsum(energy) / np.sum(energy) * 100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(cumulative_energy, 'g-', linewidth=2)
    ax.axhline(y=90, color='r', linestyle='--', 
               linewidth=1.5, label='90% Energy')
    ax.axhline(y=95, color='orange', linestyle='--', 
               linewidth=1.5, label='95% Energy')
    
    ax.set_xlabel('Number of Components (k)', fontsize=12)
    ax.set_ylabel('Cumulative Energy (%)', fontsize=12)
    ax.set_title('Energy Retention vs Rank', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    return fig

def create_comparison_plot(original, compressed, k, metrics):
    """
    Create side-by-side comparison plot
    
    Parameters:
    -----------
    original : numpy.ndarray
        Original image
    compressed : numpy.ndarray
        Compressed image
    k : int
        Rank used
    metrics : dict
        Quality metrics
    
    Returns:
    --------
    matplotlib.figure.Figure : The figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Compressed image
    axes[1].imshow(compressed, cmap='gray', vmin=0, vmax=255)
    title = f'Compressed (k={k})\nPSNR: {metrics["PSNR"]:.2f} dB'
    axes[1].set_title(title, fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig
