import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from image_compression import *

# Page configuration
st.set_page_config(
    page_title="Image Compression Tool",
    page_icon="📸",
    layout="wide"
)

# Title and description
st.title("📸 Image Compression using Linear Algebra")
st.markdown("""
This tool demonstrates **image compression** using **Singular Value Decomposition (SVD)** 
and **rank reduction** techniques from linear algebra.
""")

st.divider()

# Sidebar for controls
st.sidebar.header("⚙️ Settings")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload an Image",
    type=['png', 'jpg', 'jpeg', 'bmp'],
    help="Upload a grayscale or color image"
)

# Add example image option
use_example = st.sidebar.checkbox("Use Example Image", value=False)

# Main content
if uploaded_file is not None or use_example:
    
    # Load image
    if use_example:
        # Create a simple example image (checkerboard pattern)
        st.info("📌 Using example checkerboard image")
        size = 200
        img_matrix = np.kron(
            [[1, 0] * 4, [0, 1] * 4] * 4,
            np.ones((size//8, size//8))
        ) * 255
    else:
        img_matrix = load_image(uploaded_file, grayscale=True)
    
    if img_matrix is not None:
        
        # Get image dimensions
        m, n = img_matrix.shape
        max_rank = min(m, n)
        
        # Display original image info
        st.sidebar.success(f"✅ Image loaded: {m} × {n} pixels")
        
        # Rank slider
        k = st.sidebar.slider(
            "Select Rank (k)",
            min_value=1,
            max_value=min(100, max_rank),
            value=min(20, max_rank),
            step=1,
            help="Number of singular values to keep"
        )
        
        # Compression ratio
        compression_ratio = calculate_compression_ratio((m, n), k)
        st.sidebar.metric("Compression Ratio", f"{compression_ratio:.1f}%")
        
        # Compress button
        if st.sidebar.button("🚀 Compress Image", type="primary"):
            
            with st.spinner("Compressing image..."):
                
                # Perform SVD compression
                compressed_img = compress_image_svd(img_matrix, k)
                
                # Calculate metrics
                metrics = compute_quality_metrics(img_matrix, compressed_img)
                
                # Get singular values
                sigma = get_singular_values(img_matrix)
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Comparison",
                    "📈 Singular Values",
                    "⚡ Energy Retention",
                    "📋 Details"
                ])
                
                with tab1:
                    st.subheader("Image Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Image**")
                        st.image(img_matrix.astype(np.uint8), 
                                use_column_width=True,
                                clamp=True)
                    
                    with col2:
                        st.markdown(f"**Compressed Image (k={k})**")
                        st.image(compressed_img.astype(np.uint8),
                                use_column_width=True,
                                clamp=True)
                
                with tab2:
                    st.subheader("Singular Value Spectrum")
                    fig_sv = plot_singular_values(sigma, k)
                    st.pyplot(fig_sv)
                    plt.close()
                    
                    st.info(f"""
                    📌 **Interpretation:** 
                    - Total singular values: {len(sigma)}
                    - Using top {k} values (marked in red)
                    - Larger values contain more important information
                    """)
                
                with tab3:
                    st.subheader("Cumulative Energy Retention")
                    fig_energy = plot_energy_retention(sigma)
                    st.pyplot(fig_energy)
                    plt.close()
                    
                    # Calculate energy retained with k components
                    energy = sigma ** 2
                    energy_retained = np.sum(energy[:k]) / np.sum(energy) * 100
                    
                    st.success(f"""
                    ✅ **With k={k} components:**
                    - Retained {energy_retained:.2f}% of total energy
                    - Compression ratio: {compression_ratio:.1f}%
                    """)
                
                with tab4:
                    st.subheader("Technical Details")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📐 Matrix Dimensions")
                        st.write(f"- Original: {m} × {n}")
                        st.write(f"- Rank: {k}")
                        st.write(f"- U matrix: {m} × {k}")
                        st.write(f"- Σ matrix: {k} × {k}")
                        st.write(f"- V^T matrix: {k} × {n}")
                    
                    with col2:
                        st.markdown("### 📊 Quality Metrics")
                        st.metric("PSNR", f"{metrics['PSNR']:.2f} dB")
                        st.metric("MSE", f"{metrics['MSE']:.2f}")
                        
                        # Quality assessment
                        if metrics['PSNR'] > 40:
                            quality = "Excellent 🟢"
                        elif metrics['PSNR'] > 30:
                            quality = "Good 🟡"
                        else:
                            quality = "Fair 🟠"
                        
                        st.write(f"**Quality:** {quality}")
                    
                    st.markdown("### 💾 Storage Savings")
                    original_storage = m * n
                    compressed_storage = k * (m + n + 1)
                    
                    st.write(f"- Original: {original_storage:,} values")
                    st.write(f"- Compressed: {compressed_storage:,} values")
                    st.write(f"- Saved: {original_storage - compressed_storage:,} values")

else:
    # Instructions when no image is uploaded
    st.info("""
    👈 **Get Started:**
    1. Upload an image using the sidebar
    2. Or check "Use Example Image" to try it out
    3. Adjust the rank slider to control compression
    4. Click "Compress Image" to see results
    """)
    
    st.markdown("""
    ---
    ### 📚 About This Tool
    
    This application demonstrates **image compression** using concepts from linear algebra:
    
    - **Matrix Representation:** Images are represented as matrices
    - **SVD Decomposition:** A = U Σ V^T
    - **Rank Reduction:** Keep only top-k singular values
    - **Orthogonality:** U and V are orthogonal matrices
    - **Compression:** Reduce storage while maintaining quality
    
    **Mathematical Foundation:**
    - Vector spaces and subspaces
    - Eigenvalues and eigenvectors
    - Orthonormal bases
    - Matrix rank and dimension
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📖 MATH 201 - Linear Algebra Project | Image Compression using SVD</p>
</div>
""", unsafe_allow_html=True)
