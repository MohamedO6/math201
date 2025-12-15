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

st.title("📸 Image Compression using Singular Value Decomposition")
st.markdown("Linear Algebra Application - Matrix Rank Reduction Technique")
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
        # SVD COMPUTATION
        # ============================================================
        
        with st.spinner("Computing SVD decomposition..."):
            U, sigma, Vt = compute_svd_cached(img_matrix)
        
        st.sidebar.success("✅ SVD computed")
        
# ============================================================
# QUALITY SLIDER
# ============================================================

max_useful_k = get_max_useful_rank(m, n)
max_slider_k = min(max_rank, max_useful_k)

st.sidebar.info(f"💡 Image has {max_rank} total components")

k = st.sidebar.slider(
    "Number of Components (k)",
    min_value=1,
    max_value=max_slider_k,
    value=min(20, max_slider_k),
    step=1,
    help="Number of singular values to keep"
)

quality_percent = (k / max_rank) * 100

col_a, col_b = st.sidebar.columns(2)
col_a.metric("Quality", f"{quality_percent:.1f}%")
col_b.metric("Max useful", max_useful_k)

if k * (m + n + 1) >= m * n:
    st.sidebar.error("⚠️ No compression benefit!")

        # ============================================================
        # IMAGE COMPRESSION
        # ============================================================
        
        compressed_img = compress_image_fast(U, sigma, Vt, k)
        
        compression_ratio = calculate_compression_ratio((m, n), k)
        energy = calculate_energy_retention(sigma, k)
        metrics = compute_quality_metrics(img_matrix, compressed_img)
        
        # ============================================================
        # METRICS DISPLAY
        # ============================================================
        
        st.sidebar.divider()
        st.sidebar.metric("Compression Ratio", f"{compression_ratio:.1f}%")
        st.sidebar.metric("Energy Retained", f"{energy:.1f}%")
        st.sidebar.metric("PSNR Quality", f"{metrics['PSNR']:.1f} dB")
        
        # ============================================================
        # IMAGE COMPARISON
        # ============================================================
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Image")
            st.image(img_matrix.astype(np.uint8), width='stretch')
            st.caption(f"Storage: {m*n:,} values")
        
        with col2:
            st.markdown(f"### Compressed Image (Rank k={k})")
            st.image(compressed_img, width='stretch')
            st.caption(f"Storage: {k*(m+n+1):,} values")
        
        # ============================================================
        # SINGULAR VALUES PLOT
        # ============================================================
        
        with st.expander("📊 Singular Values Spectrum"):
            fig_sv = plot_singular_values_fast(sigma, k)
            st.pyplot(fig_sv)
            plt.close()
            st.caption("Larger singular values contain more important image information")
        
        # ============================================================
        # ENERGY RETENTION PLOT
        # ============================================================
        
        with st.expander("⚡ Energy Retention Analysis"):
            fig_energy = plot_energy_retention_fast(sigma)
            st.pyplot(fig_energy)
            plt.close()
            st.info(f"With k={k} components, {energy:.2f}% of image energy is retained")
        
        # ============================================================
        # MATHEMATICAL DETAILS
        # ============================================================
        
        with st.expander("🔬 Mathematical Details"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**SVD Decomposition: A = UΣV^T**")
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
                
                if metrics['PSNR'] > 35:
                    st.success("Quality: Excellent")
                elif metrics['PSNR'] > 25:
                    st.warning("Quality: Good")
                else:
                    st.error("Quality: Fair")
            
            st.markdown("**Storage Calculation**")
            original_storage = m * n
            compressed_storage = k * (m + n + 1)
            st.write(f"Original: {original_storage:,} values")
            st.write(f"Compressed: {compressed_storage:,} values")
            st.write(f"Savings: {original_storage - compressed_storage:,} values ({compression_ratio:.1f}%)")

# ============================================================
# WELCOME SCREEN
# ============================================================

else:
    st.info("""
    ### 🚀 Getting Started
    1. Upload an image using the sidebar
    2. Or check "Use Example Image"
    3. Adjust the quality slider to control compression
    4. View real-time results and analysis
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📐 Linear Algebra Concepts")
        st.write("• Matrix representation of images")
        st.write("• Singular Value Decomposition")
        st.write("• Orthogonal matrices and eigenvalues")
    
    with col2:
        st.markdown("### 🗜️ Compression Technique")
        st.write("• Rank-k approximation")
        st.write("• Dimension reduction")
        st.write("• Information preservation")
    
    with col3:
        st.markdown("### ⚡ Performance")
        st.write("• Cached computations")
        st.write("• Real-time compression")
        st.write("• Interactive visualization")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("📖 MATH 201 - Linear Algebra and Vector Geometry | Image Compression Project")
