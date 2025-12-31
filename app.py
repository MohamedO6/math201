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

st.title("📸 Image Compression using DCT-SVD")
st.markdown("Linear Algebra Application - DCT Transform with Low-Rank Approximation")
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
        # COMPRESSION SLIDER
        # ============================================================
        
        st.sidebar.divider()
        st.sidebar.subheader("🗜️ Compression")
        
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
        
        # ============================================================
        # COMPRESSION EXECUTION
        # ============================================================
        
        with st.spinner("Compressing image..."):
            compressed_img, sigma_vals = compress_image_dct_svd(
                img_matrix, 
                k, 
                apply_filter=apply_filter,
                sigma_filter=sigma_filter
            )
        
        compression_ratio = calculate_compression_ratio((m, n), k)
        energy = calculate_energy_retention(sigma_vals, k)
        metrics = compute_quality_metrics(img_matrix, compressed_img)
        
        # ============================================================
        # METRICS DISPLAY
        # ============================================================
        
        st.sidebar.divider()
        st.sidebar.subheader("📊 Results")
        
        col_a, col_b = st.sidebar.columns(2)
        col_a.metric("Quality", f"{quality_percent:.1f}%")
        col_b.metric("Rank k", k)
        
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
            st.markdown(f"### 🗜️ Compressed (k={k})")
            st.image(compressed_img, use_column_width=True)
            st.caption(f"Storage: {k*(m+n+1):,} values | Saved: {compression_ratio:.1f}%")
        
        # ============================================================
        # VISUALIZATIONS
        # ============================================================
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Singular Values", 
            "⚡ Energy", 
            "📉 Rank vs Error",
            "🔬 Details"
        ])
        
        with tab1:
            st.subheader("Singular Value Spectrum")
            fig_sv = plot_singular_values(sigma_vals, k)
            st.pyplot(fig_sv)
            plt.close()
            st.caption("The singular values show the importance of each component in the DCT domain")
        
        with tab2:
            st.subheader("Cumulative Energy Retention")
            fig_energy = plot_energy_retention(sigma_vals)
            st.pyplot(fig_energy)
            plt.close()
            st.info(f"✅ With k={k} components, {energy:.2f}% of image energy is retained")
        
        with tab3:
            st.subheader("Reconstruction Error vs Rank")
            with st.spinner("Computing rank-error curve..."):
                ks, errors = compute_rank_error_curve(img_matrix, max_k=min(50, max_rank))
            fig_error = plot_rank_error_curve(ks, errors)
            st.pyplot(fig_error)
            plt.close()
            st.caption("Shows how reconstruction error decreases as we use more components")
        
        with tab4:
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
                st.write(f"Original Matrix A: {m} × {n}")
                st.write(f"Rank k: {k}")
                st.write(f"Matrix U: {m} × {k}")
                st.write(f"Matrix Σ: {k} × {k}")
                st.write(f"Matrix V^T: {k} × {n}")
            
            with col_b:
                st.markdown("**Quality Metrics**")
                st.write(f"PSNR: {metrics['PSNR']:.2f} dB")
                st.write(f"MSE: {metrics['MSE']:.2f}")
                st.write(f"Energy: {energy:.2f}%")
                st.write("")
                st.markdown("**Storage Analysis**")
                original_storage = m * n
                compressed_storage = k * (m + n + 1)
                st.write(f"Original: {original_storage:,} values")
                st.write(f"Compressed: {compressed_storage:,} values")
                st.write(f"Savings: {original_storage - compressed_storage:,} values")
                st.write(f"Ratio: {compression_ratio:.1f}%")
            
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
    1. Upload an image using the sidebar
    2. Or check "Use Example Image"
    3. Optionally apply Gaussian filter for noise reduction
    4. Adjust the rank slider to control compression
    5. View real-time results and analysis
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📐 Mathematical Foundation")
        st.write("• Discrete Cosine Transform (DCT)")
        st.write("• Singular Value Decomposition")
        st.write("• Low-rank matrix approximation")
        st.write("• Energy compaction property")
    
    with col2:
        st.markdown("### 🗜️ Compression Pipeline")
        st.write("• Optional Gaussian filtering")
        st.write("• 2D DCT transformation")
        st.write("• SVD in frequency domain")
        st.write("• Rank-k truncation")
        st.write("• Inverse DCT reconstruction")
    
    with col3:
        st.markdown("### 📊 Analysis Tools")
        st.write("• Singular value spectrum")
        st.write("• Energy retention curves")
        st.write("• Rank vs error analysis")
        st.write("• PSNR quality metrics")
    
    st.divider()
    
    st.markdown("### 🎓 Linear Algebra Concepts")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        **Matrix Decomposition:**
        - DCT transforms spatial domain to frequency domain
        - SVD finds optimal orthonormal basis
        - Singular values represent importance
        - Rank reduction preserves most information
        """)
    
    with col_b:
        st.markdown("""
        **Vector Spaces:**
        - Image as vector in R^(m×n)
        - Column/row space of transformation
        - Orthogonality of singular vectors
        - Dimension reduction via truncation
        """)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("📖 MATH 201 - Linear Algebra and Vector Geometry | DCT-SVD Image Compression Project")
