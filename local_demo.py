import numpy as np
import matplotlib.pyplot as plt
from image_compression import *
import os

def run_compression_demo(image_path, k_values=[5, 10, 20, 50]):
    """
    Run image compression demo with multiple k values
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    k_values : list
        List of rank values to test
    """
    
    print("=" * 60)
    print("IMAGE COMPRESSION DEMO")
    print("=" * 60)
    
    # Load image
    print(f"\n📂 Loading image: {image_path}")
    img_matrix = load_image(image_path, grayscale=True)
    
    if img_matrix is None:
        print("❌ Error loading image!")
        return
    
    m, n = img_matrix.shape
    print(f"✅ Image loaded successfully!")
    print(f"   Dimensions: {m} × {n} pixels")
    print(f"   Original size: {m * n:,} values")
    
    # Get singular values
    print("\n🔍 Computing SVD...")
    sigma = get_singular_values(img_matrix)
    print(f"✅ SVD computed! Found {len(sigma)} singular values")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Test different compression levels
    print("\n" + "=" * 60)
    print("COMPRESSION RESULTS")
    print("=" * 60)
    
    results = []
    
    for k in k_values:
        print(f"\n🎯 Testing with k = {k}")
        print("-" * 40)
        
        # Compress image
        compressed = compress_image_svd(img_matrix, k)
        
        # Calculate metrics
        compression_ratio = calculate_compression_ratio((m, n), k)
        metrics = compute_quality_metrics(img_matrix, compressed)
        
        # Calculate energy retained
        energy = sigma ** 2
        energy_retained = np.sum(energy[:k]) / np.sum(energy) * 100
        
        # Print results
        print(f"   Compression Ratio: {compression_ratio:.1f}%")
        print(f"   PSNR: {metrics['PSNR']:.2f} dB")
        print(f"   MSE: {metrics['MSE']:.2f}")
        print(f"   Energy Retained: {energy_retained:.2f}%")
        
        # Save compressed image
        output_path = f"results/compressed_k{k}.png"
        plt.imsave(output_path, compressed, cmap='gray', vmin=0, vmax=255)
        print(f"   💾 Saved: {output_path}")
        
        results.append({
            'k': k,
            'compressed': compressed,
            'compression_ratio': compression_ratio,
            'psnr': metrics['PSNR'],
            'energy': energy_retained
        })
    
    # Create comparison plot
    print("\n📊 Creating comparison plots...")
    
    # Plot 1: All compressions side by side
    n_images = len(k_values) + 1
    fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))
    
    # Original
    axes[0].imshow(img_matrix, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Compressed versions
    for idx, result in enumerate(results):
        axes[idx + 1].imshow(result['compressed'], cmap='gray', vmin=0, vmax=255)
        title = f"k={result['k']}\nPSNR: {result['psnr']:.1f} dB"
        axes[idx + 1].set_title(title, fontsize=10)
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/comparison_all.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: results/comparison_all.png")
    plt.close()
    
    # Plot 2: Singular values
    fig_sv = plot_singular_values(sigma)
    plt.savefig('results/singular_values.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: results/singular_values.png")
    plt.close()
    
    # Plot 3: Energy retention
    fig_energy = plot_energy_retention(sigma)
    plt.savefig('results/energy_retention.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: results/energy_retention.png")
    plt.close()
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'k':<8} {'Compression':<15} {'PSNR (dB)':<12} {'Energy %':<12}")
    print("-" * 60)
    for result in results:
        print(f"{result['k']:<8} {result['compression_ratio']:<14.1f}% "
              f"{result['psnr']:<11.2f} {result['energy']:<11.2f}%")
    
    print("\n✅ Demo completed! Check the 'results' folder for output files.")
    print("=" * 60)

# ====================================
# MAIN EXECUTION
# ====================================

if __name__ == "__main__":
    
    # 🔧 USER CONFIGURATION
    # Change this to your image path
    IMAGE_PATH = "sample_image.jpg"  # <-- PUT YOUR IMAGE HERE
    
    # Choose compression levels to test
    K_VALUES = [5, 10, 20, 50, 100]  # <-- ADJUST AS NEEDED
    
    # Run the demo
    run_compression_demo(IMAGE_PATH, K_VALUES)
