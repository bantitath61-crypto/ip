import streamlit as st
import numpy as np
import cv2
from skimage import exposure
import matplotlib.pyplot as plt
from io import BytesIO

# ------------------------- Utility Functions -------------------------

def plot_histogram(img, color=False):
    fig, ax = plt.subplots()
    if len(img.shape) == 2 or not color:
        ax.hist(img.ravel(), bins=256, range=[0, 256], color='black')
    else:
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            ax.hist(img[..., i].ravel(), bins=256, range=[0, 256], color=col, alpha=0.5)
    ax.set_xlim([0, 256])
    ax.set_title('Histogram')
    ax.set_xlabel('Pixel value')
    ax.set_ylabel('Frequency')
    fig.tight_layout()
    return fig

def convert_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def show_image_and_hist(title, img, color=False):
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption=f"{title}", channels="BGR" if color else "GRAY", use_column_width=True)
    with col2:
        hist_fig = plot_histogram(img, color=color)
        st.pyplot(hist_fig)

# ------------------------- Processing Functions -------------------------

def linear_negative(img):
    return 255 - img

def contrast_stretching(img, r1, s1, r2, s2):
    def piecewise(v):
        if v < r1:
            return s1 / r1 * v
        elif v < r2:
            return ((s2 - s1) / (r2 - r1)) * (v - r1) + s1
        else:
            return ((255 - s2) / (255 - r2)) * (v - r2) + s2
    vectorized = np.vectorize(piecewise)
    return vectorized(img).astype(np.uint8)

def piecewise_linear(img, points):
    x = [pt[0] for pt in points]
    y = [pt[1] for pt in points]
    return np.interp(img, x, y).astype(np.uint8)

def log_transform(img, c):
    img_float = img.astype(np.float32)
    return (c * np.log1p(img_float)).clip(0, 255).astype(np.uint8)

def gamma_transform(img, gamma, c=1):
    img_float = img.astype(np.float32) / 255.0
    return (c * (img_float ** gamma) * 255).clip(0, 255).astype(np.uint8)

def hist_equalization(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def adaptive_hist_eq(img, clip_limit=0.01):
    return exposure.equalize_adapthist(img, clip_limit=clip_limit) * 255

def clahe_transform(img, clip_limit, tile_grid_size):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    if len(img.shape) == 2:
        return clahe.apply(img)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ------------------------- Streamlit UI -------------------------

st.set_page_config(page_title="Advanced Image Processing", layout="wide")
st.title("🧠 Advanced Image Processing Techniques")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
mode = st.radio("Choose image type:", ['Grayscale', 'Color'], horizontal=True)

method = st.selectbox(
    "Select Processing Method",
    ['Linear Negative', 'Contrast Stretching', 'Piecewise Linear Transformation',
     'Log Transformation', 'Gamma Transformation',
     'Histogram Equalization', 'Adaptive Histogram Equalization', 'CLAHE']
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert if grayscale is selected
    is_gray = (mode == 'Grayscale')
    if is_gray:
        img = convert_to_grayscale(img)

    st.subheader("Original Image & Histogram")
    show_image_and_hist("Original", img, color=not is_gray)

    # ---------- Parameter Selection ----------
    st.subheader(f"🔧 Parameters for {method}")
    if method == 'Linear Negative':
        processed = linear_negative(img)

    elif method == 'Contrast Stretching':
        r1 = st.slider("r1", 0, 255, 50)
        s1 = st.slider("s1", 0, 255, 0)
        r2 = st.slider("r2", 0, 255, 200)
        s2 = st.slider("s2", 0, 255, 255)
        processed = contrast_stretching(img, r1, s1, r2, s2)

    elif method == 'Piecewise Linear Transformation':
        st.markdown("Define control points (x, y) for interpolation:")
        points = []
        for i in range(3):
            col1, col2 = st.columns(2)
            with col1:
                x = st.slider(f"Input x{i+1}", 0, 255, i * 100)
            with col2:
                y = st.slider(f"Output y{i+1}", 0, 255, i * 100)
            points.append((x, y))
        points = sorted(points, key=lambda p: p[0])
        processed = piecewise_linear(img, points)

    elif method == 'Log Transformation':
        c = st.slider("Multiplier (c)", 1, 50, 10)
        processed = log_transform(img, c)

    elif method == 'Gamma Transformation':
        gamma = st.slider("Gamma", 0.1, 5.0, 1.0)
        c = st.slider("Multiplier (c)", 1, 10, 1)
        processed = gamma_transform(img, gamma, c)

    elif method == 'Histogram Equalization':
        processed = hist_equalization(img)

    elif method == 'Adaptive Histogram Equalization':
        clip_limit = st.slider("Clip Limit", 0.001, 0.1, 0.01)
        img_float = img.astype(np.float32) / 255.0
        processed = adaptive_hist_eq(img_float, clip_limit).astype(np.uint8)

    elif method == 'CLAHE':
        clip_limit = st.slider("Clip Limit", 1.0, 10.0, 2.0)
        tile_grid_size = st.slider("Tile Grid Size", 1, 16, 8)
        processed = clahe_transform(img, clip_limit, tile_grid_size)

    # ---------- Display Result ----------
    st.subheader("🖼️ Processed Image & Histogram")
    show_image_and_hist("Processed", processed, color=not is_gray)
