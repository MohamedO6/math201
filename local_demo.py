import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.linalg import svd
from scipy.fft import dct, idct
from scipy.ndimage import gaussian_filter
import os

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def dct2(matrix):
    return dct(dct(matrix.T, norm='ortho').T, norm='ortho')

def idct2(matrix):
    return idct(idct(matrix.T, norm='ortho').T, norm='ortho')

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

def calculate_energy_retention(sigma, k):
    energy = sigma ** 2
    return np.sum(energy[:k]) / np.sum(energy) * 100

# ============================================================
# MAIN DEMO FUNCTION
# ============================================================

def run_compression_demo(image_path, k_values=[5, 10, 20, 50], apply_filter=False, sigma_filter=1.0):
    
    print("=" * 70)
    print("DCT-SVD IMAGE COMPRESSION DEMO")
    print("=" * 70)
    
    # ============================================================
    # IMAGE LOADING
    # ============================================================
    
    print(f"\n📂 Loading image: {image_path}")
    
    img = Image.open(image_path).convert('L')
    img_matrix = np.array(img, dtype=np.float32)
    
    m, n = img_matrix.shape
    print(f"✅ Image loaded successfully!")
    print(f"   Dimensions: {m} × {n} pixels")
    print(f"   Original size: {m * n:,} values")
    
    if apply_filter:
        print(f"\n🔧 Applying Gaussian filter (σ={sigma_filter})...")
        img_matrix_filtered = gaussian_filter(img_matrix, sigma=sigma_filter)
        print(f"✅ Filter applied")
    else:
        print(f"\n🔧 No filter applied")
        img_matrix_filtered = img_matrix
    
    # ============================================================
    # DCT COMPUTATION
    # ============================================================
    
    print("\n🔍 Computing 2D DCT and SVD...")
    dct_matrix = dct2(img_matrix_filtered)
    U, sigma, Vt = svd(dct_matrix, full_matrices=False)
    print(f"✅ DCT-SVD computed! Found {len(sigma)} singular values")
    print(f"   Top 5 singular values: {sigma[:5]}")
    
    # ============================================================
    # RESULTS DIRECTORY
    # ============================================================
    
    os.makedirs("results", exist_ok=True)
    
    # ============================================================
    # COMPRESSION TESTS
    # ============================================================
    
    print("\n" + "=" * 70)
    print("COMPRESSION RESULTS")
    print("=" * 70)
    
    results = []
    
    for k in k_values:
        print(f"\n🎯 Testing with k = {k}")
        print("-" * 50)
        
        compressed, _ = compress_image_dct_svd(img_matrix_filtered, k, apply_filter=False)
        
        compression_ratio = calculate_compression_ratio((m, n), k)
        
        mse = np.mean((img_matrix.astype(float) - compressed.astype(float)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        energy_retained = calculate_energy_retention(sigma, k)
        
        print(f"   Compression Ratio: {compression_ratio:.1f}%")
        print(f"   PSNR: {psnr:.2f} dB")
        print(f"   MSE: {mse:.2f}")
        print(f"   Energy Retained: {energy_retained:.2f}%")
        
        output_path = f"results/compressed_k{k}.png"
        plt.imsave(output_path, compressed, cmap='gray', vmin=0, vmax=255)
        print(f"   💾 Saved: {output_path}")
        
        results.append({
            'k': k,
            'compressed': compressed,
            'compression_ratio': compression_ratio,
            'psnr': psnr,
            'energy': energy_retained
        })
    
    # ============================================================
    # COMPARISON PLOT
    # ============================================================
    
    print("\n📊 Creating comparison plots...")
    
    n_images = len(k_values) + 1
    fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))
    
    axes[0].imshow(img_matrix, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    for idx, result in enumerate(results):
        axes[idx + 1].imshow(result['compressed'], cmap='gray', vmin=0, vmax=255)
        title = f"k={result['k']}\nPSNR: {result['psnr']:.1f} dB"
        axes[idx + 1].set_title(title, fontsize=10)
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/comparison_all.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: results/comparison_all.png")
    plt.close()
    
    # ============================================================
    # SINGULAR VALUES PLOT
    # ============================================================
    
    fig, ax = plt.subplots(figsize=(10, 5))
    n_plot = min(100, len(sigma))
    ax.plot(range(n_plot), sigma[:n_plot], 'b-', linewidth=2)
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Singular Value', fontsize=12)
    ax.set_title('Singular Value Spectrum (DCT Domain)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/singular_values.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: results/singular_values.png")
    plt.close()
    
    # ============================================================
    # ENERGY RETENTION PLOT
    # ============================================================
    
    energy = sigma ** 2
    cumulative_energy = np.cumsum(energy) / np.sum(energy) * 100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    n_plot = min(100, len(cumulative_energy))
    ax.plot(range(n_plot), cumulative_energy[:n_plot], 'g-', linewidth=2)
    ax.axhline(y=90, color='r', linestyle='--', linewidth=1.5, label='90% Energy')
    ax.axhline(y=95, color='orange', linestyle='--', linewidth=1.5, label='95% Energy')
    ax.set_xlabel('Rank (k)', fontsize=12)
    ax.set_ylabel('Energy Retained (%)', fontsize=12)
    ax.set_title('Cumulative Energy Retention', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig('results/energy_retention.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: results/energy_retention.png")
    plt.close()
    
    # ============================================================
    # RANK VS ERROR CURVE
    # ============================================================
    
    print("\n📉 Computing rank vs error curve...")
    
    max_k_plot = min(50, len(sigma))
    errors = []
    
    for k in range(1, max_k_plot + 1):
        dct_approx = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
        reconstructed = idct2(dct_approx)
        error = np.linalg.norm(img_matrix_filtered - reconstructed, ord='fro')
        errors.append(error)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, max_k_plot + 1), errors, 'b-', linewidth=2)
    ax.set_xlabel('Rank (k)', fontsize=12)
    ax.set_ylabel('Frobenius Norm Error', fontsize=12)
    ax.set_title('Reconstruction Error vs Rank', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/rank_error_curve.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: results/rank_error_curve.png")
    plt.close()
    
    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Rank k':<10} {'Compression':<15} {'PSNR (dB)':<12} {'Energy %':<12}")
    print("-" * 70)
    for result in results:
        print(f"{result['k']:<10} {result['compression_ratio']:<14.1f}% "
              f"{result['psnr']:<11.2f} {result['energy']:<11.2f}%")
    
    print("\n✅ Demo completed! Check the 'results' folder for output files.")
    print("=" * 70)
    
    print("\n📁 Generated files:")
    print("   • comparison_all.png - Side-by-side comparison")
    print("   • singular_values.png - Singular value spectrum")
    print("   • energy_retention.png - Energy retention curve")
    print("   • rank_error_curve.png - Error vs rank analysis")
    print("   • compressed_k*.png - Individual compressed images")

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    # 🔧 USER CONFIGURATION
    IMAGE_PATH = "sample_image.jpg"
    
    K_VALUES = [5, 10, 20, 50, 100]
    
    APPLY_FILTER = False
    SIGMA_FILTER = 1.0
    
    # Run the demo
    run_compression_demo(
        IMAGE_PATH, 
        K_VALUES,
        apply_filter=APPLY_FILTER,
        sigma_filter=SIGMA_FILTER
    )
