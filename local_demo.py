import numpy as np
import matplotlib.pyplot as plt
from image_compression import *
import os

# ============================================================
# MAIN DEMO FUNCTION
# ============================================================

def run_compression_demo(image_path, k_values=[5, 10, 20, 50]):
    
    print("=" * 60)
    print("IMAGE COMPRESSION DEMO - SVD APPLICATION")
    print("=" * 60)
    
    # ============================================================
    # IMAGE LOADING
    # ============================================================
    
    print(f"\n📂 Loading image: {image_path}")
    
    img = Image.open(image_path).convert('L')
    img_matrix = np.array(img, dtype=np.float32)
    
    if img_matrix is None:
        print("❌ Error loading image!")
        return
    
    m, n = img_matrix.shape
    print(f"✅ Image loaded successfully!")
    print(f"   Dimensions: {m} × {n} pixels")
    print(f"   Original size: {m * n:,} values")
    
    # ============================================================
    # SVD COMPUTATION
    # ============================================================
    
    print("\n🔍 Computing Singular Value Decomposition...")
    from scipy.linalg import svd
    U, sigma, Vt = svd(img_matrix, full_matrices=False)
    print(f"✅ SVD computed! Found {len(sigma)} singular values")
    
    # ============================================================
    # RESULTS DIRECTORY
    # ============================================================
    
    os.makedirs("results", exist_ok=True)
    
# ============================================================
# COMPRESSION TESTS
# ============================================================

# Filter k values to only useful ones
m, n = img_matrix.shape
max_useful_k = int((m * n) / (m + n + 1)) - 1
k_values_filtered = [k for k in k_values if k <= max_useful_k]

if len(k_values_filtered) < len(k_values):
    print(f"\n⚠️  Warning: Some k values removed (max useful k = {max_useful_k})")
    print(f"   Using k values: {k_values_filtered}")

for k in k_values_filtered:
    # ... rest of code
    
    compression_ratio = calculate_compression_ratio((m, n), k)
    
    # Skip if no compression benefit
    if compression_ratio <= 0:
        print(f"   ⚠️  Skipped - no compression benefit")
        continue

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
    ax.plot(sigma, 'b-', linewidth=2)
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Singular Value', fontsize=12)
    ax.set_title('Singular Value Spectrum', fontsize=14, fontweight='bold')
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
    ax.plot(cumulative_energy, 'g-', linewidth=2)
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
    # SUMMARY TABLE
    # ============================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Rank k':<10} {'Compression':<15} {'PSNR (dB)':<12} {'Energy %':<12}")
    print("-" * 60)
    for result in results:
        print(f"{result['k']:<10} {result['compression_ratio']:<14.1f}% "
              f"{result['psnr']:<11.2f} {result['energy']:<11.2f}%")
    
    print("\n✅ Demo completed! Check the 'results' folder for output files.")
    print("=" * 60)

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    IMAGE_PATH = "sample_image.jpg"
    
    K_VALUES = [5, 10, 20, 50, 100]
    
    run_compression_demo(IMAGE_PATH, K_VALUES)
