import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import svd, dct, idct
import streamlit as st
from scipy.ndimage import gaussian_filter

# ============================================================
# IMAGE LOADING & PREPROCESSING
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
# OPTIONAL LINEAR FILTERING
# ============================================================

def apply_gaussian_filter(image_matrix, sigma=1):
    return gaussian_filter(image_matrix, sigma=sigma)

# ============================================================
# 2D DCT / IDCT
# ============================================================

def dct2(image_matrix):
    return dct(dct(image_matrix.T, norm='ortho').T, norm='ortho')

def idct2(freq_matrix):
    return idct(idct(freq_matrix.T, norm='ortho').T, norm='ortho')

# ============================================================
# SVD COMPUTATION
# ============================================================

@st.cache_data
def compute_svd_cached(matrix):
    U, sigma, Vt = svd(matrix, full_matrices=False)
    return U, sigma, Vt

# ============================================================
# LOW-RANK APPROXIMATION
# ============================================================

def compress_image_dct_svd(image_matrix, k, apply_filter=False, sigma_filter=1):
    # Step 1: Optional filtering
    if apply_filter:
        image_matrix = apply_gaussian_filter(image_matrix, sigma=sigma_filter)
    
    # Step 2: 2D DCT
    dct_matrix = dct2(image_matrix)
    
    # Step 3: SVD on DCT
    U, S, Vt = compute_svd_cached(dct_matrix)
    
    # Step 4: Keep top-k singular values
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    
    # Step 5: Low-rank approximation in DCT domain
    dct_k = U_k @ S_k @ Vt_k
    
    # Step 6: Reconstruct image via inverse DCT
    compressed_image = idct2(dct_k)
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
    
    return compressed_image, S

# ============================================================
# METRICS
# ============================================================

def calculate_compression_ratio(original_shape, k):
    m, n = original_shape
    original_size = m * n
    compressed_size = k * (m + n + 1)
    if compressed_size >= original_size:
        return 0.0
    return (1 - compressed_size / original_size) * 100

def compute_quality_metrics(original, compressed):
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')
    return {'MSE': mse, 'PSNR': psnr}

def calculate_energy_retention(sigma, k):
    energy = sigma ** 2
    return np.sum(energy[:k]) / np.sum(energy) * 100

# ============================================================
# VISUALIZATION
# ============================================================

def plot_singular_values(sigma, k=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    n_plot = min(100, len(sigma))
    ax.plot(range(n_plot), sigma[:n_plot], 'b-', linewidth=2)
    if k is not None and k < len(sigma):
        ax.axvline(x=k, color='r', linestyle='--', linewidth=2, label=f'k = {k}')
        ax.plot(k, sigma[k], 'ro', markersize=8)
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Value Spectrum')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_energy_retention(sigma):
    fig, ax = plt.subplots(figsize=(8, 4))
    energy = sigma ** 2
    cumulative_energy = np.cumsum(energy) / np.sum(energy) * 100
    n_plot = min(100, len(cumulative_energy))
    ax.plot(range(n_plot), cumulative_energy[:n_plot], 'g-', linewidth=2)
    ax.axhline(y=90, color='r', linestyle='--', linewidth=1.5, label='90% Energy')
    ax.axhline(y=95, color='orange', linestyle='--', linewidth=1.5, label='95% Energy')
    ax.set_xlabel('Rank (k)')
    ax.set_ylabel('Energy Retained (%)')
    ax.set_title('Cumulative Energy Retention')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 105])
    plt.tight_layout()
    return fig

# ============================================================
# RANK VS ERROR
# ============================================================

def compute_rank_error_curve(image_matrix, max_k=50):
    dct_matrix = dct2(image_matrix)
    U, S, Vt = compute_svd_cached(dct_matrix)
    errors = []
    ks = range(1, min(max_k, len(S)) + 1)
    for k in ks:
        S_k = np.diag(S[:k])
        U_k = U[:, :k]
        Vt_k = Vt[:k, :]
        approx = idct2(U_k @ S_k @ Vt_k)
        error = np.linalg.norm(image_matrix - approx, ord='fro')
        errors.append(error)
    return ks, errors

def plot_rank_error_curve(ks, errors):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, errors, 'b-', linewidth=2)
    ax.set_xlabel('Rank (k)')
    ax.set_ylabel('Frobenius Norm Error')
    ax.set_title('Error vs Rank')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
