import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import svd
import streamlit as st

# ============= OPTIMIZED FUNCTIONS =============

@st.cache_data  # Cache image loading
def load_image_optimized(image_file, grayscale=True, max_size=800):
    """
    Load and optimize image with automatic resizing
    
    Parameters:
    -----------
    image_file : file object or str
        Uploaded file or path
    grayscale : bool
        Convert to grayscale
    max_size : int
        Maximum dimension (auto-resize for speed)
    
    Returns:
    --------
    numpy.ndarray : Optimized image matrix
    """
    try:
        img = Image.open(image_file)
        
        # Auto-resize for performance
        if grayscale:
            img = img.convert('L')
        
        # Resize if too large
        width, height = img.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        return np.array(img, dtype=np.float32)  # float32 faster than float64
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

@st.cache_data  # Cache SVD computation
def compute_svd_cached(image_matrix):
    """
    Compute SVD with caching (expensive operation)
    """
    U, sigma, Vt = svd(image_matrix, full_matrices=False)
    return U, sigma, Vt

def compress_image_fast(U, sigma, Vt, k):
    """
    Fast compression using pre-computed SVD
    
    Parameters:
    -----------
    U, sigma, Vt : numpy arrays
        Pre-computed SVD components
    k : int
        Rank to use
    
    Returns:
    --------
    numpy.ndarray : Compressed image
    """
    # Just slice the pre-computed arrays
    compressed = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
    return np.clip(compressed, 0, 255).astype(np.uint8)

def calculate_compression_ratio(original_shape, k):
    """Calculate compression ratio"""
    m, n = original_shape
    original_size = m * n
    compressed_size = k * (m + n + 1)
    ratio = (1 - compressed_size / original_size) * 100
    return ratio

@st.cache_data
def compute_quality_metrics(original, compressed):
    """Compute MSE and PSNR with caching"""
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    return {'MSE': mse, 'PSNR': psnr}

@st.cache_data
def calculate_energy_retention(sigma, k):
    """Calculate energy retained with caching"""
    energy = sigma ** 2
    energy_retained = np.sum(energy[:k]) / np.sum(energy) * 100
    return energy_retained

# ============= PLOTTING FUNCTIONS =============

def plot_singular_values_fast(sigma, k=None):
    """Fast plotting with minimal processing"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot only first 100 values for speed
    n_plot = min(100, len(sigma))
    ax.plot(range(n_plot), sigma[:n_plot], 'b-', linewidth=2)
    
    if k is not None and k <= n_plot:
        ax.axvline(x=k, color='r', linestyle='--', linewidth=2, label=f'k = {k}')
        ax.plot(k, sigma[k], 'ro', markersize=8)
    
    ax.set_xlabel('Index', fontsize=11)
    ax.set_ylabel('Singular Value', fontsize=11)
    ax.set_title('Singular Value Spectrum (First 100)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_energy_retention_fast(sigma):
    """Fast energy plot"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Calculate cumulative energy
    energy = sigma ** 2
    cumulative_energy = np.cumsum(energy) / np.sum(energy) * 100
    
    # Plot only first 100 for speed
    n_plot = min(100, len(cumulative_energy))
    ax.plot(range(n_plot), cumulative_energy[:n_plot], 'g-', linewidth=2)
    
    ax.axhline(y=90, color='r', linestyle='--', linewidth=1.5, label='90% Energy')
    ax.axhline(y=95, color='orange', linestyle='--', linewidth=1.5, label='95% Energy')
    
    ax.set_xlabel('Rank (k)', fontsize=11)
    ax.set_ylabel('Energy Retained (%)', fontsize=11)
    ax.set_title('Cumulative Energy (First 100 components)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    return fig
