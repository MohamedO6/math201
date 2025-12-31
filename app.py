import streamlit as st
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from image_compression import *

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Image Compression Tool",
    page_icon="📸",
    layout="wide"
)

st.title("📸 Image Compression using Linear Algebra")
st.markdown("DCT-SVD and Frequency Domain Methods")
st.divider()

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Image",
    type=['png', 'jpg', 'jpeg'],
    help="Select an image file"
)

max_dimension = st.sidebar.select_slider(
    "Maximum Image Dimension",
    options=[400, 600, 800, 1000],
    value=800,
    help="Larger images take longer to process"
)

use_example = st.sidebar.checkbox("Use Example Image")

# ============================================================
# IMAGE LOADING
# ============================================================

if uploaded_file is not None or use_example:
    
    if use_example:
        st.info("📌 Using example checkerboard pattern")
        size = 400
        img_matrix = np.kron(
            [[1, 0] * 4, [0, 1] * 4] * 4,
            np.ones((size//8, size//8))
        ) * 255
        img_matrix = img_matrix.astype(np.float32)
    else:
        with st.spinner("Loading image..."):
            img_matrix = load_image_optimized(uploaded_file, grayscale=True, max_size=max_dimension)
    
    if img_matrix is not None:
        m, n = img_matrix.shape
        max_rank = min(m, n)
        
        st.sidebar.success(f"✅ Image: {m} × {n} pixels")
        
        # ============================================================
        # FILTER OPTIONS
        # ============================================================
        
        st.sidebar.divider()
        st.sidebar.subheader("🔧 Preprocessing")
        
        apply_filter = st.sidebar.checkbox(
            "Apply Gaussian Filter",
            value=False,
            help="Reduces noise before compression"
        )
        
        if apply_filter:
            sigma_filter = st.sidebar.slider(
                "Filter Strength (σ)",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.5,
                help="Higher values = more smoothing"
            )
        else:
            sigma_filter = 1.0
        
        # ============================================================
        # COMPRESSION METHOD
        # ============================================================
        
        st.sidebar.divider()
        st.sidebar.subheader("📐 Compression Method")
        
        compression_method = st.sidebar.radio(
            "Select Method",
            ["DCT-SVD (Rank)", "Frequency Masking"],
            help="Choose compression algorithm"
        )
        
        if compression_method == "DCT-SVD (Rank)":
            
            st.sidebar.info(f"💡 Image has {max_rank} total components")
            
            k = st.sidebar.slider(
                "Number of Components (k)",
                min_value=1,
                max_value=min(max_rank, 200),
                value=min(20, max_rank),
                step=1,
                help="Number of singular values to keep"
            )
            
            quality_percent = (k / max_rank) * 100
            
            # Compression
            with st.spinner("Compressing image..."):
                compressed_img, sigma_vals = compress_image_dct_svd(
                    img_matrix, 
                    k, 
                    apply_filter=apply_filter,
                    sigma_filter=sigma_filter
                )
            
            compression_ratio = calculate_compression_ratio((m, n), k)
            energy = calculate_energy_retention(sigma_vals, k)
            
        else:  # Frequency Masking
            
            keep_fraction = st.sidebar.slider(
                "Keep Fraction",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Fraction of frequency components to keep"
            )
            
            # Compression
            with st.spinner("Compressing image..."):
                if apply_filter:
                    img_to_compress = apply_gaussian_filter(img_matrix, sigma=sigma_filter)
                else:
                    img_to_compress = img_matrix
                
                compressed_img = partial_reconstruction(img_to_compress, keep_fraction)
            
            compression_ratio = calculate_freq_compression_ratio((m, n), keep_fraction)
            sigma_vals = np.ones(10)
            energy = keep_fraction * 100
            quality_percent = keep_fraction * 100
        
        metrics = compute_quality_metrics(img_matrix, compressed_img)
        
        # ============================================================
        # METRICS DISPLAY
        # ============================================================
        
        st.sidebar.divider()
        st.sidebar.subheader("📊 Results")
        
        if compression_method == "DCT-SVD (Rank)":
            col_a, col_b = st.sidebar.columns(2)
            col_a.metric("Quality", f"{quality_percent:.1f}%")
            col_b.metric("Rank k", k)
        else:
            st.sidebar.metric("Keep Fraction", f"{keep_fraction:.2f}")
        
        st.sidebar.metric("Compression Ratio", f"{compression_ratio:.1f}%")
        st.sidebar.metric("Energy Retained", f"{energy:.1f}%")
        st.sidebar.metric("PSNR Quality", f"{metrics['PSNR']:.1f} dB")
        
        if metrics['PSNR'] > 35:
            st.sidebar.success("🟢 Excellent Quality")
        elif metrics['PSNR'] > 25:
            st.sidebar.warning("🟡 Good Quality")
        else:
            st.sidebar.error("🟠 Fair Quality")
        
        # ============================================================
        # IMAGE COMPARISON
        # ============================================================
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 Original Image")
            st.image(img_matrix.astype(np.uint8), use_column_width=True)
            st.caption(f"Storage: {m*n:,} values")
        
        with col2:
            if compression_method == "DCT-SVD (Rank)":
                st.markdown(f"### 🗜️ Compressed (k={k})")
            else:
                st.markdown(f"### 🗜️ Compressed (frac={keep_fraction:.2f})")
            st.image(compressed_img, use_column_width=True)
            st.caption(f"PSNR: {metrics['PSNR']:.1f} dB | Saved: {compression_ratio:.1f}%")
        
        # ============================================================
        # VISUALIZATIONS
        # ============================================================
        
        if compression_method == "DCT-SVD (Rank)":
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Singular Values", 
                "⚡ Energy", 
                "📉 Rank vs Error",
                "🌊 Frequency Spectrum",
                "🔬 Details"
            ])
            
            with tab1:
                st.subheader("Singular Value Spectrum")
                fig_sv = plot_singular_values(sigma_vals, k)
                st.pyplot(fig_sv)
                plt.close()
                st.caption("The singular values show the importance of each component")
            
            with tab2:
                st.subheader("Cumulative Energy Retention")
                fig_energy = plot_energy_retention(sigma_vals)
                st.pyplot(fig_energy)
                plt.close()
                st.info(f"✅ With k={k} components, {energy:.2f}% of energy retained")
            
            with tab3:
                st.subheader("Reconstruction Error vs Rank")
                with st.spinner("Computing rank-error curve..."):
                    ks, errors = compute_rank_error_curve(img_matrix, max_k=min(50, max_rank))
                fig_error = plot_rank_error_curve(ks, errors)
                st.pyplot(fig_error)
                plt.close()
                st.caption("Error decreases as we use more components")
            
            with tab4:
                st.subheader("Frequency Spectrum")
                fig_spectrum = plot_frequency_spectrum(img_matrix)
                st.pyplot(fig_spectrum)
                plt.close()
                st.caption("Low frequencies (center) contain most image information")
            
            with tab5:
                st.subheader("Mathematical Details")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**DCT-SVD Decomposition**")
                    st.write("1. Apply 2D DCT transform")
                    st.write("2. Compute SVD: DCT(A) = UΣV^T")
                    st.write("3. Keep top-k singular values")
                    st.write("4. Reconstruct via inverse DCT")
                    st.write("")
                    st.markdown("**Matrix Dimensions**")
                    st.write(f"Original: {m} × {n}")
                    st.write(f"Rank: {k}")
                    st.write(f"U: {m} × {k}")
                    st.write(f"Σ: {k} × {k}")
                    st.write(f"V^T: {k} × {n}")
                
                with col_b:
                    st.markdown("**Quality Metrics**")
                    st.write(f"PSNR: {metrics['PSNR']:.2f} dB")
                    st.write(f"MSE: {metrics['MSE']:.2f}")
                    st.write(f"Energy: {energy:.2f}%")
                    st.write("")
                    st.markdown("**Storage**")
                    original_storage = m * n
                    compressed_storage = k * (m + n + 1)
                    st.write(f"Original: {original_storage:,}")
                    st.write(f"Compressed: {compressed_storage:,}")
                    st.write(f"Ratio: {compression_ratio:.1f}%")
        
        else:  # Frequency Masking
            tab1, tab2 = st.tabs(["🌊 Frequency Spectrum", "🔬 Details"])
            
            with tab1:
                st.subheader("Frequency Spectrum")
                fig_spectrum = plot_frequency_spectrum(img_matrix)
                st.pyplot(fig_spectrum)
                plt.close()
                st.caption(f"Keeping {keep_fraction*100:.0f}% of frequency components (centered)")
            
            with tab2:
                st.subheader("Mathematical Details")
                
                st.markdown("**Frequency Masking Method**")
                st.write("1. Apply 2D FFT transform")
                st.write("2. Shift zero-frequency to center")
                st.write("3. Keep only low-frequency components")
                st.write("4. Reconstruct via inverse FFT")
                
                st.divider()
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Parameters**")
                    st.write(f"Keep Fraction: {keep_fraction:.2f}")
                    st.write(f"Image Size: {m} × {n}")
                    kept_size = int(m * keep_fraction)
                    st.write(f"Mask Size: {kept_size} × {kept_size}")
                
                with col_b:
                    st.markdown("**Quality**")
                    st.write(f"PSNR: {metrics['PSNR']:.2f} dB")
                    st.write(f"MSE: {metrics['MSE']:.2f}")
                    st.write(f"Compression: {compression_ratio:.1f}%")

# ============================================================
# WELCOME SCREEN
# ============================================================

else:
    st.info("""
    ### 🚀 Getting Started
    1. Upload an image or use example
    2. Choose compression method
    3. Adjust parameters
    4. View results and analysis
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📐 DCT-SVD Method")
        st.write("• 2D Discrete Cosine Transform")
        st.write("• Singular Value Decomposition")
        st.write("• Rank-k approximation")
        st.write("• Optimal low-rank representation")
    
    with col2:
        st.markdown("### 🌊 Frequency Masking")
        st.write("• 2D Fourier Transform")
        st.write("• Low-pass frequency filtering")
        st.write("• Spatial domain reconstruction")
        st.write("• Preserves low frequencies")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("📖 MATH 201 - Linear Algebra | Image Compression Project")
