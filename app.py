import streamlit as st
import numpy as np
from PIL import Image
import matplotlib
from io import BytesIO
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
        
        # Initialize variables
        compressed_img = None
        sigma_vals = None
        compression_ratio = 0
        energy = 0
        quality_percent = 0
        
        if compression_method == "DCT-SVD (Rank)":
            
            st.sidebar.info(f"💡 Image has {max_rank} total components")
            
            # ============================================================
            # COMPUTE RECOMMENDATION
            # ============================================================
            
            with st.spinner("Computing recommendations..."):
                # Apply DCT first
                if apply_filter:
                    img_for_svd = apply_gaussian_filter(img_matrix, sigma=sigma_filter)
                else:
                    img_for_svd = img_matrix
                
                dct_temp = dct2(img_for_svd)
                _, sigma_temp, _ = compute_svd_cached(dct_temp)
                
                # Get recommendation
                rec_k, rec_energy, rec_reason = recommend_k(sigma_temp, max_rank, target_energy=0.90)
            
            # ============================================================
            # RECOMMENDATION SECTION
            # ============================================================
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 💡 Smart Recommendation")
            
            col_rec1, col_rec2 = st.sidebar.columns([2, 1])
            with col_rec1:
                st.metric("Recommended k", rec_k)
            with col_rec2:
                if st.button("Use", key="use_rec_k"):
                    st.session_state.recommended_k = rec_k
            
            st.sidebar.caption(f"📊 {rec_reason}")
            st.sidebar.caption(f"🗜️ Compression: {calculate_compression_ratio((m, n), rec_k):.1f}%")
            
            st.sidebar.markdown("---")
            
            # ============================================================
            # K SLIDER
            # ============================================================
            
            # Check if we should use recommended k
            if 'recommended_k' in st.session_state:
                default_k = st.session_state.recommended_k
                del st.session_state.recommended_k  # Use only once
            else:
                default_k = min(20, max_rank)
            
            k = st.sidebar.slider(
                "Number of Components (k)",
                min_value=1,
                max_value=min(max_rank, 200),
                value=default_k,
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
            
            # Detailed reconstruction analysis
            recon_analysis = analyze_reconstruction_quality(
                img_matrix, 
                compressed_img, 
                k, 
                sigma_vals
            )
            
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
            energy = keep_fraction * 100
            quality_percent = keep_fraction * 100
            
            # Dummy values for compatibility
            recon_analysis = None
        
        # Calculate metrics (works for both methods)
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
        # DOWNLOAD SECTION
        # ============================================================
        
        st.divider()
        
        col_a, col_b, col_c = st.columns([1, 2, 1])
        
        with col_b:
            st.markdown("### 📥 Download Compressed Image")
            
            # Create download buffer
            buf = BytesIO()
            Image.fromarray(compressed_img).save(buf, format='PNG')
            byte_data = buf.getvalue()
            
            # Calculate file size
            file_size_kb = len(byte_data) / 1024
            
            # Download button
            if compression_method == "DCT-SVD (Rank)":
                filename = f"compressed_k{k}.png"
            else:
                filename = f"compressed_frac{keep_fraction:.2f}.png"
            
            st.download_button(
                label="📥 Download PNG",
                data=byte_data,
                file_name=filename,
                mime="image/png",
                use_container_width=True
            )
            
            # Show file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", f"{file_size_kb:.1f} KB")
            with col2:
                st.metric("Compression", f"{compression_ratio:.1f}%")
            with col3:
                st.metric("Quality", f"{metrics['PSNR']:.1f} dB")
        
        st.divider()

        # ============================================================
        # VISUALIZATIONS
        # ============================================================
        
        if compression_method == "DCT-SVD (Rank)":
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Singular Values", 
                "⚡ Energy", 
                "📉 Rank vs Error",
                "🔄 Reconstruction",
                "🌊 Frequency Spectrum",
                "🔬 Details"
            ])
            
            with tab1:
                st.subheader("Singular Value Spectrum")
                try:
                    fig_sv = plot_singular_values(sigma_vals, k)
                    st.pyplot(fig_sv)
                    plt.close()
                    st.caption("The singular values show the importance of each component in the DCT domain")
                    
                    st.info(f"""
                    **Interpretation:**
                    - **Top {k} components** are kept (left of red line)
                    - **Remaining {len(sigma_vals)-k} components** are discarded
                    - Large singular values = important image features
                    - Small singular values = noise or fine details
                    """)
                except Exception as e:
                    st.error(f"Error plotting singular values: {e}")
            
            with tab2:
                st.subheader("Cumulative Energy Retention")
                try:
                    fig_energy = plot_energy_retention(sigma_vals)
                    st.pyplot(fig_energy)
                    plt.close()
                    st.info(f"✅ With k={k} components, {energy:.2f}% of energy retained")
                    
                    st.markdown("""
                    **Energy Analysis:**
                    - Energy = (Singular Value)²
                    - Most energy in first few components
                    - 90% energy typically achieved with few components
                    """)
                except Exception as e:
                    st.error(f"Error plotting energy: {e}")
            
            with tab3:
                st.subheader("Reconstruction Error vs Rank")
                try:
                    with st.spinner("Computing rank-error curve..."):
                        ks, errors = compute_rank_error_curve(img_matrix, max_k=min(50, max_rank))
                    fig_error = plot_rank_error_curve(ks, errors)
                    st.pyplot(fig_error)
                    plt.close()
                    st.caption("Error decreases as we use more components")
                    
                    st.markdown("""
                    **Error Behavior:**
                    - Error measured in Frobenius norm
                    - Decreases monotonically with k
                    - Trade-off between quality and compression
                    """)
                except Exception as e:
                    st.error(f"Error computing rank-error curve: {e}")
            
            with tab4:
                st.subheader("Reconstruction Analysis")
                try:
                    fig_recon = plot_reconstruction_comparison(img_matrix, compressed_img, k)
                    st.pyplot(fig_recon)
                    plt.close()
                    
                    st.markdown("### 📐 Reconstruction Pipeline")
                    st.write("""
                    **Step-by-Step Process:**
                    
                    1. **Forward DCT:** Spatial Domain → Frequency Domain
                       - Transform image to cosine basis
                       - Y = DCT(A)
                    
                    2. **SVD Decomposition:** Find Principal Components
                       - Y = U × Σ × V^T
                       - Singular values sorted by importance
                    
                    3. **Low-Rank Approximation:** Keep Top-k
                       - Y_k = U_k × Σ_k × V_k^T
                       - Discard small singular values
                    
                    4. **Inverse DCT:** Frequency Domain → Spatial Domain
                       - A_reconstructed = IDCT(Y_k)
                       - Back to pixel representation
                    
                    5. **Post-processing:** Clip to [0, 255]
                       - Ensure valid pixel values
                    """)
                    
                    if recon_analysis:
                        st.markdown("### 📊 Detailed Metrics")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Energy Retained", f"{recon_analysis['energy_retained']:.2f}%")
                            st.metric("Frobenius Error", f"{recon_analysis['frobenius_error']:.2f}")
                        with col_b:
                            st.metric("Relative Error", f"{recon_analysis['relative_error']:.2f}%")
                            st.metric("Original Storage", f"{recon_analysis['original_storage']:,}")
                        with col_c:
                            st.metric("Compressed Storage", f"{recon_analysis['compressed_storage']:,}")
                            st.metric("Compression Ratio", f"{recon_analysis['compression_ratio']:.1f}%")
                    
                except Exception as e:
                    st.error(f"Error in reconstruction analysis: {e}")
            
            with tab5:
                st.subheader("Frequency Spectrum")
                try:
                    fig_spectrum = plot_frequency_spectrum(img_matrix)
                    st.pyplot(fig_spectrum)
                    plt.close()
                    st.caption("Low frequencies (center) contain most image information")
                    
                    st.markdown("""
                    **Frequency Domain:**
                    - **Bright center** = low frequencies (general structure)
                    - **Darker edges** = high frequencies (fine details)
                    - DCT concentrates energy in top-left
                    """)
                except Exception as e:
                    st.error(f"Error plotting spectrum: {e}")
            
            with tab6:
                st.subheader("Mathematical Details")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**DCT-SVD Decomposition**")
                    st.write("**Forward Transform:**")
                    st.latex(r"Y = \text{DCT}(A) = C \times A \times C^T")
                    st.write("")
                    st.write("**SVD:**")
                    st.latex(r"Y = U \times \Sigma \times V^T")
                    st.write("")
                    st.write("**Low-Rank Approximation:**")
                    st.latex(r"Y_k = U_k \times \Sigma_k \times V_k^T")
                    st.write("")
                    st.write("**Inverse Transform:**")
                    st.latex(r"A_{rec} = \text{IDCT}(Y_k) = C^T \times Y_k \times C")
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
                    
                    if recon_analysis:
                        st.write(f"Frobenius Error: {recon_analysis['frobenius_error']:.2f}")
                        st.write(f"Relative Error: {recon_analysis['relative_error']:.2f}%")
                    
                    st.write("")
                    st.markdown("**Storage Analysis**")
                    original_storage = m * n
                    compressed_storage = k * (m + n + 1)
                    st.write(f"Original: {original_storage:,}")
                    st.write(f"Compressed: {compressed_storage:,}")
                    st.write(f"Savings: {original_storage - compressed_storage:,}")
                    st.write(f"Ratio: {compression_ratio:.1f}%")
                
                st.divider()
                
                st.markdown("**Key Properties**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **Orthonormality:**
                    - C^T × C = I
                    - U^T × U = I
                    - V^T × V = I
                    """)
                with col2:
                    st.markdown("""
                    **Optimality:**
                    - Eckart-Young theorem
                    - Minimizes ||Y - Y_k||_F
                    - Best rank-k approximation
                    """)
                
                st.divider()
                
                if apply_filter:
                    st.info(f"🔧 Gaussian filter applied with σ={sigma_filter}")
                else:
                    st.info("🔧 No preprocessing filter applied")
        
        else:  # Frequency Masking
            tab1, tab2 = st.tabs(["🌊 Frequency Spectrum", "🔬 Details"])
            
            with tab1:
                st.subheader("Frequency Spectrum")
                try:
                    fig_spectrum = plot_frequency_spectrum(img_matrix)
                    st.pyplot(fig_spectrum)
                    plt.close()
                    st.caption(f"Keeping {keep_fraction*100:.0f}% of frequency components (centered)")
                    
                    st.markdown("""
                    **Frequency Masking Method:**
                    - Apply FFT to transform to frequency domain
                    - Keep only low-frequency components (center)
                    - Discard high-frequency components (edges)
                    - Apply IFFT to reconstruct image
                    """)
                except Exception as e:
                    st.error(f"Error plotting spectrum: {e}")
            
            with tab2:
                st.subheader("Mathematical Details")
                
                st.markdown("**Frequency Masking Method**")
                st.write("**Forward Transform:**")
                st.latex(r"Y = \text{FFT}(A)")
                st.write("**Masking:**")
                st.latex(r"Y_{masked} = Y \odot M")
                st.write("(M is binary mask keeping center frequencies)")
                st.write("")
                st.write("**Inverse Transform:**")
                st.latex(r"A_{rec} = |\text{IFFT}(Y_{masked})|")
                
                st.divider()
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Parameters**")
                    st.write(f"Keep Fraction: {keep_fraction:.2f}")
                    st.write(f"Image Size: {m} × {n}")
                    kept_size = int(m * keep_fraction)
                    st.write(f"Mask Size: {kept_size} × {kept_size}")
                    st.write(f"Kept Components: {int((keep_fraction**2)*100)}%")
                
                with col_b:
                    st.markdown("**Quality**")
                    st.write(f"PSNR: {metrics['PSNR']:.2f} dB")
                    st.write(f"MSE: {metrics['MSE']:.2f}")
                    st.write(f"Compression: {compression_ratio:.1f}%")
                
                st.divider()
                
                if apply_filter:
                    st.info(f"🔧 Gaussian filter applied with σ={sigma_filter}")
                else:
                    st.info("🔧 No preprocessing filter applied")

# ============================================================
# WELCOME SCREEN
# ============================================================

else:
    st.info("""
    ### 🚀 Getting Started
    1. Upload an image or use example
    2. Choose compression method (DCT-SVD or Frequency Masking)
    3. Adjust parameters to control compression
    4. View real-time results and detailed analysis
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📐 DCT-SVD Method")
        st.write("**Pipeline:**")
        st.write("1. Forward DCT (Spatial → Frequency)")
        st.write("2. SVD Decomposition")
        st.write("3. Low-Rank Approximation (keep top-k)")
        st.write("4. Inverse DCT (Frequency → Spatial)")
        st.write("")
        st.write("**Features:**")
        st.write("• Optimal rank-k approximation")
        st.write("• Energy-based compression")
        st.write("• Mathematically principled")
        st.write("• Excellent quality/size trade-off")
    
    with col2:
        st.markdown("### 🌊 Frequency Masking")
        st.write("**Pipeline:**")
        st.write("1. Forward FFT (Spatial → Frequency)")
        st.write("2. Apply mask (keep low frequencies)")
        st.write("3. Inverse FFT (Frequency → Spatial)")
        st.write("")
        st.write("**Features:**")
        st.write("• Simple and fast")
        st.write("• Low-pass filtering")
        st.write("• Good for smooth images")
        st.write("• Natural noise reduction")
    
    st.divider()
    
    st.markdown("### 🎓 Linear Algebra Concepts")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("**Change of Basis**")
        st.write("• DCT/FFT as basis transformation")
        st.write("• Orthonormal matrices")
        st.write("• Frequency representation")
        st.write("• Reversible transformations")
    
    with col_b:
        st.markdown("**SVD Decomposition**")
        st.write("• Matrix factorization")
        st.write("• Singular values & vectors")
        st.write("• Rank reduction")
        st.write("• Optimal approximation")
    
    with col_c:
        st.markdown("**Reconstruction**")
        st.write("• Inverse transforms")
        st.write("• Energy preservation")
        st.write("• Error minimization")
        st.write("• Quality metrics")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("📖 MATH 201 - Linear Algebra and Vector Geometry | Image Compression Project")
