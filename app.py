import streamlit as st
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from image_compression import *

# Page config
st.set_page_config(
    page_title="Fast Image Compressor",
    page_icon="⚡",
    layout="wide"
)

# Title
st.title("⚡ Fast Image Compression using SVD")
st.markdown("**Optimized with caching for instant results!**")
st.divider()

# Sidebar
st.sidebar.header("⚙️ Controls")

# Upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Image",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a grayscale or color image"
)

# Max size option
max_dimension = st.sidebar.select_slider(
    "Image Size (for speed)",
    options=[400, 600, 800, 1000],
    value=800,
    help="Larger = better quality but slower"
)

use_example = st.sidebar.checkbox("Use Example Image")

# Main app
if uploaded_file is not None or use_example:
    
    # Load image
    if use_example:
        st.info("📌 Using example checkerboard pattern")
        size = 400
        img_matrix = np.kron(
            [[1, 0] * 4, [0, 1] * 4] * 4,
            np.ones((size//8, size//8))
        ) * 255
        img_matrix = img_matrix.astype(np.float32)
    else:
        with st.spinner("⏳ Loading image..."):
            img_matrix = load_image_optimized(uploaded_file, grayscale=True, max_size=max_dimension)
    
    if img_matrix is not None:
        m, n = img_matrix.shape
        max_rank = min(m, n)
        
        st.sidebar.success(f"✅ Loaded: {m}×{n} pixels")
        
        # Compute SVD ONCE (cached)
        with st.spinner("🔄 Computing SVD (one-time)..."):
            U, sigma, Vt = compute_svd_cached(img_matrix)
        
        st.sidebar.success("✅ SVD computed!")
        
        # Rank slider
        k = st.sidebar.slider(
            "Compression Level (Rank k)",
            min_value=1,
            max_value=min(100, max_rank),
            value=min(20, max_rank),
            step=1,
            help="Adjust in real-time!"
        )
        
        # Real-time compression (FAST)
        compressed_img = compress_image_fast(U, sigma, Vt, k)
        
        # Metrics
        compression_ratio = calculate_compression_ratio((m, n), k)
        energy = calculate_energy_retention(sigma, k)
        metrics = compute_quality_metrics(img_matrix, compressed_img)
        
        # Display metrics in sidebar
        st.sidebar.metric("Compression", f"{compression_ratio:.1f}%")
        st.sidebar.metric("Energy Retained", f"{energy:.1f}%")
        st.sidebar.metric("PSNR", f"{metrics['PSNR']:.1f} dB")
        
        # Main display - side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 Original Image")
            st.image(img_matrix.astype(np.uint8), use_container_width=True, clamp=True)
            st.caption(f"Size: {m}×{n} = {m*n:,} values")
        
        with col2:
            st.markdown(f"### 🗜️ Compressed (k={k})")
            st.image(compressed_img, use_container_width=True, clamp=True)
            st.caption(f"Storage: {k*(m+n+1):,} values ({compression_ratio:.1f}% savings)")
        
        # Expandable sections
        with st.expander("📊 View Singular Values"):
            fig_sv = plot_singular_values_fast(sigma, k)
            st.pyplot(fig_sv)
            plt.close()
        
        with st.expander("⚡ View Energy Retention"):
            fig_energy = plot_energy_retention_fast(sigma)
            st.pyplot(fig_energy)
            plt.close()
            st.info(f"💡 With k={k}, you retain {energy:.2f}% of the image information!")
        
        with st.expander("🔬 Technical Details"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Matrix Dimensions:**")
                st.write(f"- Original: {m} × {n}")
                st.write(f"- Rank: {k}")
                st.write(f"- U: {m} × {k}")
                st.write(f"- Σ: {k} × {k}")
                st.write(f"- V^T: {k} × {n}")
            
            with col_b:
                st.markdown("**Quality Metrics:**")
                st.write(f"- PSNR: {metrics['PSNR']:.2f} dB")
                st.write(f"- MSE: {metrics['MSE']:.2f}")
                
                if metrics['PSNR'] > 35:
                    quality = "Excellent 🟢"
                elif metrics['PSNR'] > 25:
                    quality = "Good 🟡"
                else:
                    quality = "Fair 🟠"
                
                st.write(f"- Quality: {quality}")

else:
    # Welcome screen
    st.info("""
    ### 🚀 Get Started:
    1. Upload an image from the sidebar
    2. Or use the example image
    3. Adjust the slider to see **instant** compression!
    
    **⚡ Optimized with caching** - SVD computed once, compressions are instant!
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📐 Matrix Theory")
        st.write("Images = Matrices")
        st.write("SVD: A = UΣV^T")
    
    with col2:
        st.markdown("### 🗜️ Compression")
        st.write("Rank reduction")
        st.write("Keep top-k values")
    
    with col3:
        st.markdown("### ⚡ Performance")
        st.write("Cached computations")
        st.write("Real-time updates")

# Footer
st.divider()
st.caption("📖 MATH 201 - Linear Algebra | Optimized Image Compression")
