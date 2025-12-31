import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.linalg import svd
from scipy.fft import dct, idct, fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
import os

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def dct2(matrix):
    return dct(dct(matrix.T, norm='ortho').T, norm='ortho')

def idct2(matrix):
    return idct(idct(matrix.T, norm='ortho').T, norm='ortho')

def partial_reconstruction(image, keep_fraction=0.5):
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

def compress_image_dct_svd(image_matrix, k, apply_filter=False, sigma_filter=1):
    if apply_filter:
        image_matrix = gaussian_filter(image_matrix, sigma=sigma_filter)
    
    dct_matrix = dct2(image_matrix)
    U, S, Vt = svd(dct_matrix, full_matrices=False)
    
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    
    dct_k = U_k @ S_k @ Vt_k
    compressed_image = idct2(dct_k)
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
    
    return compressed_image, S

def calculate_compression_ratio(original_shape, k):
    m, n = original_shape
    original_size = m * n
    compressed_size = k * (m + n + 1)
    if compressed_size >= original_size:
        return 0.0
    return (1 - compressed_size / original_size) * 100

def calculate_freq_compression_ratio(original_shape, keep_fraction):
    m, n = original_shape
    total_components = m * n
    kept_components = int(total_components * keep_fraction * keep_fraction)
    ratio = (1 - kept_components / total_components) * 100
    return max(0.0, ratio)

def calculate_energy_retention(sigma, k):
    energy = sigma ** 2
    return np.sum(energy[:k]) / np.sum(energy) * 100

# ============================================================
# DCT-SVD DEMO
# ============================================================

def run_compression_demo(image_path, k_values=[5, 10, 20, 50], apply_filter=False, sigma_filter=1.0):
    
    print("=" * 70)
    print("DCT-SVD IMAGE COMPRESSION DEMO")
    print("=" * 70)
    
    print(f"\n📂 Loading image: {image_path}")
    
    img = Image.open(image_path).convert('L')
    img_matrix = np.array(img, dtype=np.float32)
    
    m, n = img_matrix.shape
    print(f"✅ Image loaded: {m} × {n} pixels")
    
    if apply_filter:
        print(f"\n🔧 Applying Gaussian filter (σ={sigma_filter})...")
        img_matrix_filtered = gaussian_filter(img_matrix, sigma=sigma_filter)
    else:
        img_matrix_filtered = img_matrix
    
    print("\n🔍 Computing DCT and SVD...")
    dct_matrix = dct2(img_matrix_filtered)
    U, sigma, Vt = svd(dct_matrix, full_matrices=False)
    print(f"✅ SVD computed! {len(sigma)} singular values")
    
    os.makedirs("results", exist_ok=True)
    
    print("\n" + "=" * 70)
    print("COMPRESSION RESULTS")
    print("=" * 70)
    
    results = []
    
    for k in k_values:
        print(f"\n🎯 k = {k}")
        print("-" * 50)
        
        compressed, _ = compress_image_dct_svd(img_matrix_filtered, k)
        
        compression_ratio = calculate_compression_ratio((m, n), k)
        
        mse = np.mean((img_matrix.astype(float) - compressed.astype(float)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        energy_retained = calculate_energy_retention(sigma, k)
        
        print(f"   Compression: {compression_ratio:.1f}%")
        print(f"   PSNR: {psnr:.2f} dB")
        print(f"   Energy: {energy_retained:.2f}%")
        
        output_path = f"results/dct_svd_k{k}.png"
        plt.imsave(output_path, compressed, cmap='gray', vmin=0, vmax=255)
        print(f"   💾 {output_path}")
        
        results.append({
            'k': k,
            'compressed': compressed,
            'compression_ratio': compression_ratio,
            'psnr': psnr
        })
    
    # Comparison plot
    n_images = len(k_values) + 1
    fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))
    
    axes[0].imshow(img_matrix, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original', fontweight='bold')
    axes[0].axis('off')
    
    for idx, result in enumerate(results):
        axes[idx + 1].imshow(result['compressed'], cmap='gray', vmin=0, vmax=255)
        axes[idx + 1].set_title(f"k={result['k']}\n{result['psnr']:.1f} dB")
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/dct_svd_comparison.png', dpi=150, bbox_inches='tight')
    print("\n   ✅ results/dct_svd_comparison.png")
    plt.close()
    
    print("\n✅ DCT-SVD demo completed!")

# ============================================================
# FREQUENCY MASKING DEMO
# ============================================================

def run_frequency_masking_demo(image_path, keep_fractions=[0.1, 0.3, 0.5, 0.7, 0.9]):
    
    print("\n" + "=" * 70)
    print("FREQUENCY MASKING COMPRESSION DEMO")
    print("=" * 70)
    
    print(f"\n📂 Loading image: {image_path}")
    img = Image.open(image_path).convert('L')
    img_matrix = np.array(img, dtype=np.float32)
    
    m, n = img_matrix.shape
    print(f"✅ Image loaded: {m} × {n} pixels")
    
    os.makedirs("results_freq", exist_ok=True)
    
    print("\n" + "=" * 70)
    print("FREQUENCY MASKING RESULTS")
    print("=" * 70)
    
    results = []
    
    for fraction in keep_fractions:
        print(f"\n🎯 Fraction = {fraction:.2f}")
        print("-" * 50)
        
        compressed = partial_reconstruction(img_matrix, fraction)
        
        compression_ratio = calculate_freq_compression_ratio((m, n), fraction)
        
        mse = np.mean((img_matrix.astype(float) - compressed.astype(float)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        print(f"   Compression: {compression_ratio:.1f}%")
        print(f"   PSNR: {psnr:.2f} dB")
        
        output_path = f"results_freq/freq_mask_{fraction:.2f}.png"
        plt.imsave(output_path, compressed, cmap='gray', vmin=0, vmax=255)
        print(f"   💾 {output_path}")
        
        results.append({
            'fraction': fraction,
            'compressed': compressed,
            'psnr': psnr
        })
    
    # Comparison plot
    n_images = len(keep_fractions) + 1
    fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))
    
    axes[0].imshow(img_matrix, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original', fontweight='bold')
    axes[0].axis('off')
    
    for idx, result in enumerate(results):
        axes[idx + 1].imshow(result['compressed'], cmap='gray', vmin=0, vmax=255)
        axes[idx + 1].set_title(f"f={result['fraction']:.2f}\n{result['psnr']:.1f} dB")
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results_freq/freq_comparison.png', dpi=150, bbox_inches='tight')
    print("\n   ✅ results_freq/freq_comparison.png")
    plt.close()
    
    print("\n✅ Frequency masking demo completed!")

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    IMAGE_PATH = "sample_image.jpg"
    
    # Part 1: DCT-SVD
    print("\n" + "="*70)
    print("PART 1: DCT-SVD COMPRESSION")
    print("="*70)
    run_compression_demo(IMAGE_PATH, [5, 10, 20, 50, 100])
    
    # Part 2: Frequency Masking
    print("\n\n" + "="*70)
    print("PART 2: FREQUENCY MASKING COMPRESSION")
    print("="*70)
    run_frequency_masking_demo(IMAGE_PATH, [0.1, 0.3, 0.5, 0.7, 0.9])
    
    print("\n" + "="*70)
    print("ALL DEMOS COMPLETED!")
    print("="*70)
    print("\n📁 Check folders:")
    print("   • results/ - DCT-SVD compression results")
    print("   • results_freq/ - Frequency masking results")
