import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.fft import dct, idct, fft2, ifft2, fftshift, ifftshift
import streamlit as st
from scipy.ndimage import gaussian_filter

# ============================================================
# IMAGE LOADING & PREPROCESSING
# ============================================================

@st.cache_data
def load_image_optimized(image_file, grayscale=True, max_size=800):
    """Load and resize image efficiently"""
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
    """Apply Gaussian smoothing filter"""
    return gaussian_filter(image_matrix, sigma=sigma)

# ============================================================
# 2D DCT / IDCT
# ============================================================

def dct2(image_matrix):
    """2D Discrete Cosine Transform"""
    return dct(dct(image_matrix.T, norm='ortho').T, norm='ortho')

def idct2(freq_matrix):
    """2D Inverse Discrete Cosine Transform"""
    return idct(idct(freq_matrix.T, norm='ortho').T, norm='ortho')

# ============================================================
# FREQUENCY DOMAIN MASKING
# ============================================================

def partial_reconstruction(image, keep_fraction=0.5):
    """
    Reconstruct image using a fraction of frequency components.
    Keeps low-frequency components centered around DC.
    """
    dft = fft2(image)
    dft_shifted = fftshift(dft)
    rows, cols = dft_shifted.shape
    center_row, center_col = rows // 2, cols // 2
    
    # Create rectangular mask
    mask = np.zeros_like(dft_shifted, dtype=bool)
    row_start = int(center_row - keep_fraction * rows // 2)
    row_end = int(center_row + keep_fraction * rows // 2)
    col_start = int(center_col - keep_fraction * cols // 2)
    col_end = int(center_col + keep_fraction * cols // 2)
    
    mask[row_start:row_end, col_start:col_end] = True
    dft_shifted *= mask
    
    dft_inverse = ifft2(ifftshift(dft_shifted))
    return np.abs(dft_inverse).astype(np.uint8)

def calculate_freq_compression_ratio(original_shape, keep_fraction):
    """Calculate compression ratio for frequency masking"""
    m, n = original_shape
    total_components = m * n
    kept_components = int(total_components * keep_fraction * keep_fraction)
    ratio = (1 - kept_components / total_components) * 100
    return max(0.0, ratio)

# ============================================================
# SVD COMPUTATION
# ============================================================

@st.cache_data
def compute_svd_cached(matrix):
    """Compute SVD with caching for performance"""
    U, sigma, Vt = svd(matrix, full_matrices=False)
    return U, sigma, Vt

# ============================================================
# DCT-SVD COMPRESSION
# ============================================================

def compress_image_dct_svd(image_matrix, k, apply_filter=False, sigma_filter=1):
    """
    Compress image using DCT-SVD method.
    
    Steps:
    1. Optional Gaussian filtering
    2. Apply 2D DCT
    3. Compute SVD
    4. Keep top-k singular values
    5. Reconstruct via inverse DCT
    """
    # Step 1: Optional filtering
    if apply_filter:
        image_matrix = apply_gaussian_filter(image_matrix, sigma=sigma_filter)
    
    # Step 2: DCT
    dct_matrix = dct2(image_matrix)
    
    # Step 3: SVD
    U, S, Vt = compute_svd_cached(dct_matrix)
    
    # Step 4: Low-rank approximation
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    
    dct_k = U_k @ S_k @ Vt_k
    
    # Step 5: Inverse DCT
    compressed_image = idct2(dct_k)
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
    
    return compressed_image, S

# ============================================================
# METRICS
# ============================================================

def get_max_useful_rank(m, n):
    """Calculate maximum useful rank for compression"""
    max_k = int((m * n) / (m + n + 1)) - 1
    return max(1, max_k)

def calculate_compression_ratio(original_shape, k):
    """Calculate compression ratio for DCT-SVD method"""
    m, n = original_shape
    original_size = m * n
    compressed_size = k * (m + n + 1)
    if compressed_size >= original_size:
        return 0.0
    return (1 - compressed_size / original_size) * 100

def compute_quality_metrics(original, compressed):
    """Compute MSE and PSNR quality metrics"""
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return {'MSE': mse, 'PSNR': psnr}

def calculate_energy_retention(sigma, k):
    """Calculate percentage of energy retained with k components"""
    energy = sigma ** 2
    total_energy = np.sum(energy)
    if total_energy == 0:
        return 0.0
    return np.sum(energy[:k]) / total_energy * 100

# ============================================================
# VISUALIZATION
# ============================================================

def plot_singular_values(sigma, k=None):
    """Plot singular value spectrum"""
    fig, ax = plt.subplots(figsize=(8, 4))
    n_plot = min(100, len(sigma))
    ax.plot(range(n_plot), sigma[:n_plot], 'b-', linewidth=2)
    if k is not None and k < len(sigma):
        ax.axvline(x=k, color='r', linestyle='--', linewidth=2, label=f'k = {k}')
        ax.plot(k, sigma[k], 'ro', markersize=8)
    ax.set_xlabel('Index', fontsize=11)
    ax.set_ylabel('Singular Value', fontsize=11)
    ax.set_title('Singular Value Spectrum', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_energy_retention(sigma):
    """Plot cumulative energy retention curve"""
    fig, ax = plt.subplots(figsize=(8, 4))
    energy = sigma ** 2
    total_energy = np.sum(energy)
    if total_energy == 0:
        cumulative_energy = np.zeros(len(sigma))
    else:
        cumulative_energy = np.cumsum(energy) / total_energy * 100
    
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

def plot_frequency_spectrum(image_matrix):
    """Plot frequency spectrum magnitude"""
    fig, ax = plt.subplots(figsize=(8, 6))
    dft = fft2(image_matrix)
    dft_shifted = fftshift(dft)
    magnitude = np.log(np.abs(dft_shifted) + 1)
    
    im = ax.imshow(magnitude, cmap='hot')
    ax.set_title('Frequency Spectrum (Log Magnitude)', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig

# ============================================================
# RANK VS ERROR
# ============================================================

def compute_rank_error_curve(image_matrix, max_k=50):
    """Compute reconstruction error for different ranks"""
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
    return list(ks), errors

def plot_rank_error_curve(ks, errors):
    """Plot reconstruction error vs rank"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, errors, 'b-', linewidth=2)
    ax.set_xlabel('Rank (k)', fontsize=11)
    ax.set_ylabel('Frobenius Norm Error', fontsize=11)
    ax.set_title('Reconstruction Error vs Rank', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
