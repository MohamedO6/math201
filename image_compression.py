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
# 2D DCT / IDCT (CHANGE OF BASIS)
# ============================================================

def dct2(image_matrix):
    """
    2D Discrete Cosine Transform - Forward Change of Basis
    
    Transforms image from spatial domain to frequency domain:
    Y = C × A × C^T
    
    where:
    - A is the original image (spatial domain)
    - C is the DCT basis matrix (orthonormal cosine basis)
    - Y is the transformed image (frequency domain)
    
    Mathematical Properties:
    - DCT is orthonormal: C^T × C = I
    - Energy compaction: most information in few coefficients
    - Reversible: can apply inverse to get back original
    """
    return dct(dct(image_matrix.T, norm='ortho').T, norm='ortho')

def idct2(freq_matrix):
    """
    2D Inverse Discrete Cosine Transform - Reverse Change of Basis
    
    Transforms from frequency domain back to spatial domain:
    A_reconstructed = C^T × Y_k × C
    
    where:
    - Y_k is the compressed frequency representation (after SVD)
    - C^T is the inverse DCT basis (transpose of C)
    - A_reconstructed is the reconstructed image (spatial domain)
    
    Key Points:
    - Reverses the DCT transformation
    - Each frequency coefficient becomes spatial intensity
    - Orthonormality ensures accurate reconstruction
    - Result is in pixel domain (viewable image)
    """
    return idct(idct(freq_matrix.T, norm='ortho').T, norm='ortho')

# ============================================================
# FREQUENCY DOMAIN MASKING
# ============================================================

def partial_reconstruction(image, keep_fraction=0.5):
    """
    Reconstruct image using frequency masking (FFT-based)
    Keeps low-frequency components centered around DC
    """
    dft = fft2(image)
    dft_shifted = fftshift(dft)
    rows, cols = dft_shifted.shape
    center_row, center_col = rows // 2, cols // 2
    
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
# DCT-SVD COMPRESSION WITH DETAILED RECONSTRUCTION
# ============================================================

def compress_image_dct_svd(image_matrix, k, apply_filter=False, sigma_filter=1):
    """
    DCT-SVD Image Compression with Full Reconstruction Pipeline
    
    COMPRESSION PIPELINE:
    =====================
    
    Step 1: PREPROCESSING (Optional)
    ---------------------------------
    Apply Gaussian filter to reduce noise:
    A_filtered = GaussianFilter(A)
    
    Step 2: FORWARD DCT (Change of Basis to Frequency Domain)
    ----------------------------------------------------------
    Transform image from spatial to frequency domain:
    Y = DCT(A) = C × A × C^T
    
    Properties:
    - Y contains frequency coefficients
    - Low frequencies (top-left) = general structure
    - High frequencies (bottom-right) = fine details
    - Energy concentrated in few coefficients
    
    Step 3: SVD DECOMPOSITION (Find Principal Components)
    -----------------------------------------------------
    Decompose frequency matrix into singular components:
    Y = U × Σ × V^T
    
    where:
    - U: m×m orthonormal matrix (left singular vectors)
    - Σ: m×n diagonal matrix (singular values, sorted descending)
    - V^T: n×n orthonormal matrix (right singular vectors)
    
    Mathematical Insight:
    - Columns of U and V are orthonormal basis vectors
    - Singular values σ_i measure importance of each component
    - Large σ_i = important features (keep these!)
    - Small σ_i = noise or minor details (discard these!)
    
    Step 4: LOW-RANK APPROXIMATION (Compression)
    --------------------------------------------
    Keep only top-k singular values and vectors:
    Y_k = U_k × Σ_k × V_k^T
    
    where:
    - U_k: m×k matrix (first k columns of U)
    - Σ_k: k×k diagonal matrix (top k singular values)
    - V_k^T: k×n matrix (first k rows of V^T)
    
    Why this works:
    - Top k components capture most image energy
    - Eckart-Young theorem: Y_k is optimal rank-k approximation
    - Minimizes ||Y - Y_k||_F (Frobenius norm error)
    
    Storage Savings:
    - Original: m×n values
    - Compressed: k(m+n+1) values
    - Ratio: [1 - k(m+n+1)/(m×n)] × 100%
    
    Step 5: INVERSE DCT (Reconstruction to Spatial Domain)
    ------------------------------------------------------
    Transform compressed frequency back to spatial domain:
    A_reconstructed = IDCT(Y_k) = C^T × Y_k × C
    
    What happens during IDCT:
    - Each frequency coefficient is a weight on DCT basis
    - IDCT combines these weighted basis functions
    - Result: reconstructed pixel intensities
    - Minor details lost (discarded small σ_i)
    - Main features preserved (kept large σ_i)
    
    Mathematical Properties:
    - DCT is orthonormal: C^T × C = I
    - Perfect reconstruction if k = rank(Y)
    - Lossy if k < rank(Y), but visually good if k chosen well
    - Energy preserved: ||A_reconstructed|| ≈ ||A|| if k is large
    
    Step 6: POST-PROCESSING
    -----------------------
    Ensure valid pixel values:
    A_reconstructed = clip(A_reconstructed, 0, 255)
    
    OUTPUTS:
    ========
    - Compressed image in spatial domain (viewable)
    - Singular values (for analysis)
    
    THE BIG PICTURE:
    ================
    Original (Spatial) → DCT → Frequency → SVD → Low-Rank → IDCT → Compressed (Spatial)
         A              →  Y  →  U×Σ×V^T  →  Y_k  →  A_reconstructed
    
    Trade-offs:
    - Higher k = better quality, less compression
    - Lower k = more compression, lower quality
    - Optimal k balances quality and storage
    """
    
    # ============================================================
    # STEP 1: PREPROCESSING
    # ============================================================
    if apply_filter:
        image_matrix = apply_gaussian_filter(image_matrix, sigma=sigma_filter)
    
    # ============================================================
    # STEP 2: FORWARD DCT (Spatial → Frequency)
    # ============================================================
    # Transform to frequency domain
    # Y = C × A × C^T
    dct_matrix = dct2(image_matrix)
    
    # At this point:
    # - dct_matrix contains frequency coefficients
    # - Top-left = low frequencies (general structure)
    # - Bottom-right = high frequencies (fine details)
    
    # ============================================================
    # STEP 3: SVD DECOMPOSITION
    # ============================================================
    # Decompose frequency matrix
    # Y = U × Σ × V^T
    U, S, Vt = compute_svd_cached(dct_matrix)
    
    # At this point:
    # - U, Vt are orthonormal bases in frequency domain
    # - S contains singular values (importance weights)
    # - S[0] is largest (most important component)
    # - S[-1] is smallest (least important component)
    
    # ============================================================
    # STEP 4: LOW-RANK APPROXIMATION
    # ============================================================
    # Keep only top-k components
    # Y_k = U_k × Σ_k × V_k^T
    
    S_k = np.diag(S[:k])  # k×k diagonal matrix of top k singular values
    U_k = U[:, :k]        # m×k matrix of first k left singular vectors
    Vt_k = Vt[:k, :]      # k×n matrix of first k right singular vectors
    
    # Reconstruct in frequency domain (compressed)
    dct_k = U_k @ S_k @ Vt_k
    
    # At this point:
    # - dct_k is rank-k approximation of dct_matrix
    # - Contains only k most important frequency components
    # - Discarded (rank - k) least important components
    
    # ============================================================
    # STEP 5: INVERSE DCT (Frequency → Spatial) - RECONSTRUCTION
    # ============================================================
    # Transform back to spatial domain
    # A_reconstructed = C^T × Y_k × C
    
    compressed_image = idct2(dct_k)
    
    # At this point:
    # - compressed_image is in spatial domain (pixels)
    # - Each pixel is linear combination of DCT basis functions
    # - Weighted by compressed frequency coefficients
    # - Result is viewable image, close to original
    
    # What was reconstructed:
    # - Main structure (from large singular values)
    # - Important features (from top-k components)
    
    # What was lost:
    # - Fine details (from small singular values)
    # - Noise (naturally filtered out)
    # - High-frequency textures (if k is small)
    
    # ============================================================
    # STEP 6: POST-PROCESSING
    # ============================================================
    # Ensure valid pixel range [0, 255]
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
    
    # Return both reconstructed image and singular values
    # Singular values used for analysis and visualization
    return compressed_image, S

# ============================================================
# RECONSTRUCTION ANALYSIS FUNCTIONS
# ============================================================

def analyze_reconstruction_quality(original, reconstructed, k, singular_values):
    """
    Analyze reconstruction quality after DCT-SVD compression
    
    Returns detailed metrics about reconstruction process
    """
    m, n = original.shape
    
    # Energy analysis
    energy = singular_values ** 2
    total_energy = np.sum(energy)
    kept_energy = np.sum(energy[:k])
    energy_ratio = (kept_energy / total_energy) * 100
    
    # Error analysis
    reconstruction_error = np.linalg.norm(original - reconstructed, ord='fro')
    relative_error = reconstruction_error / np.linalg.norm(original, ord='fro')
    
    # Storage analysis
    original_storage = m * n
    compressed_storage = k * (m + n + 1)
    storage_ratio = (compressed_storage / original_storage) * 100
    
    return {
        'energy_retained': energy_ratio,
        'frobenius_error': reconstruction_error,
        'relative_error': relative_error * 100,
        'original_storage': original_storage,
        'compressed_storage': compressed_storage,
        'storage_ratio': storage_ratio,
        'compression_ratio': 100 - storage_ratio
    }

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

def plot_reconstruction_comparison(original, reconstructed, k):
    """
    Visualize reconstruction quality
    Shows original, reconstructed, and difference
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Reconstructed
    axes[1].imshow(reconstructed, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Reconstructed (k={k})', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Difference
    diff = np.abs(original.astype(float) - reconstructed.astype(float))
    im = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    return fig

# ============================================================
# RANK VS ERROR
# ============================================================

def compute_rank_error_curve(image_matrix, max_k=50):
    """
    Compute reconstruction error for different ranks
    Shows how error decreases as k increases
    """
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
