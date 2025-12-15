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
    
    print("\n" + "=" * 60)
    print("COMPRESSION RESULTS")
    print("=" * 60)
    
    results = []
    
    for k in k_values:
        print(f"\n🎯 Testing with k = {k}")
        print("-" * 40)
        
        compressed = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
        compressed = np.clip(compressed, 0, 255).astype(np.uint8)
        
        compression_ratio = calculate_compression_ratio((m, n), k)
        
        mse = np.mean((img_matrix.astype(float) - compressed.astype(float)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        energy = sigma ** 2
        energy_retained = np.sum(energy[:k]) / np.sum(energy) * 100
        
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
