import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import svd
import streamlit as st

# ============================================================
# IMAGE LOADING
# ============================================================

@st.cache_data
def load_image_optimized(image_file, grayscale=True, max_size=800):
    try:
        img = Image.open(image_file)
        
        if grayscale:
            img = img.convert('L')
        
        width, height = img.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        return np.array(img, dtype=np.float32)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# ============================================================
# SVD COMPUTATION
# ============================================================

@st.cache_data
def compute_svd_cached(image_matrix):
    U, sigma, Vt = svd(image_matrix, full_matrices=False)
    return U, sigma, Vt

# ============================================================
# IMAGE COMPRESSION
# ============================================================

def compress_image_fast(U, sigma, Vt, k):
    compressed = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
    return np.clip(compressed, 0, 255).astype(np.uint8)

# ============================================================
# COMPRESSION METRICS
# ============================================================

def calculate_compression_ratio(original_shape, k):
    m, n = original_shape
    original_size = m * n
    compressed_size = k * (m + n + 1)
    
    if compressed_size >= original_size:
        return 0.0
    
    ratio = (1 - compressed_size / original_size) * 100
    return ratio

def get_max_useful_rank(m, n):
    """
    Calculate maximum k where compression is still beneficial
    """
    max_k = int((m * n) / (m + n + 1)) - 1
    return max(1, max_k)

# ============================================================
# VISUALIZATION: SINGULAR VALUES
# ============================================================

def plot_singular_values_fast(sigma, k=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    n_plot = min(100, len(sigma))
    ax.plot(range(n_plot), sigma[:n_plot], 'b-', linewidth=2)
    
    if k is not None and k <= n_plot:
        ax.axvline(x=k, color='r', linestyle='--', linewidth=2, label=f'k = {k}')
        ax.plot(k, sigma[k], 'ro', markersize=8)
    
    ax.set_xlabel('Index', fontsize=11)
    ax.set_ylabel('Singular Value', fontsize=11)
    ax.set_title('Singular Value Spectrum', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

# ============================================================
# VISUALIZATION: ENERGY RETENTION
# ============================================================

def plot_energy_retention_fast(sigma):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    energy = sigma ** 2
    cumulative_energy = np.cumsum(energy) / np.sum(energy) * 100
    
    n_plot = min(100, len(cumulative_energy))
    ax.plot(range(n_plot), cumulative_energy[:n_plot], 'g-', linewidth=2)
    
    ax.axhline(y=90, color='r', linestyle='--', linewidth=1.5, label='90% Energy')
    ax.axhline(y=95, color='orange', linestyle='--', linewidth=1.5, label='95% Energy')
    
    ax.set_xlabel('Rank (k)', fontsize=11)
    ax.set_ylabel('Energy Retained (%)', fontsize=11)
    ax.set_title('Cumulative Energy Retention', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    return fig
