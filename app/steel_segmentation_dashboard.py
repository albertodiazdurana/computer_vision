"""
Steel Defect Segmentation Dashboard

PURPOSE:
Interactive interface for steel defect detection using U-Net segmentation.
Demonstrates end-to-end ML pipeline from training to deployment.

WHAT IT DOES:
- Loads trained U-Net model for steel defect segmentation
- Allows selection of images from Severstal dataset
- Displays original image, ground truth mask, and prediction
- Shows per-image Dice/IoU metrics

USAGE:
    streamlit run app/steel_segmentation_dashboard.py
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Steel Defect Segmentation",
    page_icon="🔍",
    layout="wide"
)

st.title("Steel Defect Segmentation")
st.markdown("U-Net model for pixel-level defect detection on steel surfaces")

# Dataset info
with st.expander("About the Dataset"):
    st.markdown("""
**Severstal Steel Defect Detection** - [Kaggle Competition](https://www.kaggle.com/c/severstal-steel-defect-detection)

**Source:** Severstal, one of the world's leading steel and mining companies, provided this dataset
for a Kaggle competition to automate quality inspection in steel manufacturing.

**Contents:**
- **12,568 steel surface images** (256 x 1600 pixels, grayscale)
- **4 defect classes** with pixel-level annotations (Run-Length Encoded masks)
- **~53% of images contain defects**, rest are defect-free
- Defects include surface cracks, scratches, inclusions, and rolled-in scale

**Challenge:** Severe class imbalance (~3% of pixels are defects) and varied defect appearances
make this a difficult segmentation task. Competition winners achieved ~0.90 Dice using
pretrained encoders, two-stage pipelines, and extensive augmentation.

**This Dashboard:** Uses a lightweight U-Net (487K params) trained on 4,000 images,
achieving 0.42 Dice - demonstrating the core segmentation workflow within GPU memory constraints.
""")

# -----------------------------------------------------------------------------
# Paths - Use sample data for cloud deployment, full data for local development
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent

# Check if running with sample data (cloud) or full data (local)
SAMPLE_DATA_DIR = APP_DIR / "sample_data"
FULL_DATA_DIR = PROJECT_ROOT / "data" / "severstal-steel-defect-detection"

if SAMPLE_DATA_DIR.exists():
    # Cloud deployment: use bundled sample data
    DATA_DIR = SAMPLE_DATA_DIR
    MODEL_PATH = APP_DIR / "steel_unet.keras"
else:
    # Local development: use full dataset
    DATA_DIR = FULL_DATA_DIR
    MODEL_PATH = PROJECT_ROOT / "models" / "steel_unet.keras"

TRAIN_CSV = DATA_DIR / "train.csv"
IMAGE_DIR = DATA_DIR / "train_images"


# -----------------------------------------------------------------------------
# Load model (cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained U-Net model."""
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


# -----------------------------------------------------------------------------
# Load annotations (cached)
# -----------------------------------------------------------------------------
@st.cache_data
def load_annotations():
    """Load the training annotations CSV."""
    df = pd.read_csv(TRAIN_CSV)
    # Get images that have defects
    images_with_defects = df[df["EncodedPixels"].notna()]["ImageId"].unique().tolist()
    return df, images_with_defects


@st.cache_data
def compute_image_scores(_model, df, image_list, max_images=100):
    """Pre-compute Dice scores for sorting images by performance."""
    import matplotlib.pyplot as plt

    scores = []
    for img_id in image_list[:max_images]:
        # Load and preprocess image
        img_path = IMAGE_DIR / img_id
        img = plt.imread(img_path)
        if img.ndim == 2:
            img = img[..., np.newaxis]
        elif img.ndim == 3:
            img = img[..., :1]
        img_resized = tf.image.resize(img, (128, 128)).numpy() / 255.0

        # Get ground truth
        gt_mask = get_combined_mask(img_id, df)
        gt_resized = tf.image.resize(
            gt_mask[..., np.newaxis], (128, 128), method="nearest"
        ).numpy()[:, :, 0].astype(np.float32)

        # Predict
        pred = _model.predict(img_resized[np.newaxis, ...], verbose=0)
        pred_mask = (pred[0, :, :, 0] > 0.5).astype(np.float32)

        # Calculate Dice
        y_true_f = gt_resized.flatten()
        y_pred_f = pred_mask.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + 1e-6) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-6)

        scores.append((img_id, dice))

    # Sort by Dice score descending (best first)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# -----------------------------------------------------------------------------
# RLE Decoder (from notebook)
# -----------------------------------------------------------------------------
def rle_to_mask(rle_string, height=256, width=1600):
    """Convert RLE string to binary mask."""
    if pd.isna(rle_string):
        return np.zeros((height, width), dtype=np.uint8)
    
    rle = [int(x) for x in rle_string.split()]
    starts = rle[0::2]
    lengths = rle[1::2]
    
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        start -= 1  # RLE is 1-indexed
        mask[start:start + length] = 1
    
    return mask.reshape((height, width), order='F')


def get_combined_mask(image_id, df):
    """Get binary mask combining all defect classes for an image."""
    mask = np.zeros((256, 1600), dtype=np.uint8)
    image_rows = df[df["ImageId"] == image_id]
    
    for _, row in image_rows.iterrows():
        if pd.notna(row["EncodedPixels"]):
            class_mask = rle_to_mask(row["EncodedPixels"])
            mask = np.maximum(mask, class_mask)
    
    return mask


# -----------------------------------------------------------------------------
# Image preprocessing
# -----------------------------------------------------------------------------
def preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess image for model input."""
    import matplotlib.pyplot as plt
    img = plt.imread(image_path)
    
    if img.ndim == 2:
        img = img[..., np.newaxis]
    elif img.ndim == 3:
        img = img[..., :1]
    
    img_resized = tf.image.resize(img, target_size).numpy()
    img_normalized = img_resized / 255.0
    
    return img_normalized

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def calculate_dice(y_true, y_pred, smooth=1e-6):
    """Calculate Dice coefficient."""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def calculate_iou(y_true, y_pred, smooth=1e-6):
    """Calculate Intersection over Union."""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
# Load resources
model = load_model()
df, images_with_defects = load_annotations()

# Compute scores (cached - only runs once)
# Use fewer images for sample data (cloud deployment)
max_images = min(100, len(images_with_defects))
with st.spinner(f"Computing prediction scores for {max_images} images..."):
    image_scores = compute_image_scores(model, df, images_with_defects, max_images=max_images)

# Sidebar
st.sidebar.header("Image Selection")

# Dice score filter slider
min_dice = min(score for _, score in image_scores)
max_dice = max(score for _, score in image_scores)

dice_range = st.sidebar.slider(
    "Filter by Dice score:",
    min_value=float(min_dice),
    max_value=float(max_dice),
    value=(float(min_dice), float(max_dice)),
    step=0.01,
    format="%.2f",
    help="Dice score measures the overlap between predicted and ground truth masks. "
         "Range is 0-1 where 1 means perfect overlap. Use this slider to focus on "
         "best predictions (high values) or analyze failure cases (low values)."
)

# Filter images by selected range
filtered_scores = [(img_id, score) for img_id, score in image_scores
                   if dice_range[0] <= score <= dice_range[1]]

# Search by filename
search_term = st.sidebar.text_input(
    "Search by filename:",
    placeholder="e.g., 0002cc",
    help="Filter images by filename. Enter partial filename to search. "
         "Leave empty to show all images in the Dice score range."
)

# Apply search filter
if search_term:
    filtered_scores = [(img_id, score) for img_id, score in filtered_scores
                       if search_term.lower() in img_id.lower()]

st.sidebar.markdown(f"**{len(filtered_scores)}** of {len(image_scores)} images matching filters")

# Create display options with scores
image_options = [f"{img_id} (Dice: {score:.2f})" for img_id, score in filtered_scores]

# Select image
if image_options:
    selected_option = st.sidebar.selectbox(
        "Select an image:",
        image_options,
        index=0,
        help="Images are sorted by Dice score (best first). Use the filters above "
             "to narrow down the list by score range or filename."
    )
    # Extract image ID from selection
    selected_image = selected_option.split(" (Dice:")[0]
else:
    st.sidebar.warning("No images in selected range")
    selected_image = None

# Model info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader(
    "Model Info",
    help="Technical details about the trained U-Net model. The model uses an "
         "encoder-decoder architecture with skip connections, optimized for "
         "limited GPU memory (2.2GB VRAM). Trained on Severstal Steel Defect dataset."
)
st.sidebar.markdown("""
- **Architecture:** U-Net (487K params)
- **Input Size:** 128x128
- **Training:** 4,000 images
- **Val Dice:** 0.42
- **Val IoU:** 0.28
""")

# -----------------------------------------------------------------------------
# Process and Display
# -----------------------------------------------------------------------------
if selected_image:
    # Load and preprocess
    image_path = IMAGE_DIR / selected_image
    img_input = preprocess_image(image_path)
    
    # Get ground truth mask
    gt_mask = get_combined_mask(selected_image, df)
    gt_mask_resized = tf.image.resize(
        gt_mask[..., np.newaxis], (128, 128), method="nearest"
    ).numpy()[:, :, 0].astype(np.float32)
    
    # Predict
    pred = model.predict(img_input[np.newaxis, ...], verbose=0)
    pred_mask = (pred[0, :, :, 0] > 0.5).astype(np.float32)
    
    # Calculate metrics
    dice = calculate_dice(gt_mask_resized, pred_mask)
    iou = calculate_iou(gt_mask_resized, pred_mask)
    
    # Display metrics
    st.subheader(
        "Segmentation Metrics",
        help="Quantitative measures of how well the model's prediction matches the "
             "ground truth annotation. These metrics are calculated per-image and "
             "may differ from the overall validation metrics shown in Model Info."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Dice Coefficient", f"{dice:.3f}",
        help="Also known as F1 score for segmentation. Measures the overlap between "
             "prediction and ground truth: 2*(intersection)/(sum of areas). "
             "Range 0-1, where 1 is perfect overlap. More forgiving than IoU."
    )
    col2.metric(
        "IoU", f"{iou:.3f}",
        help="Intersection over Union (Jaccard Index). Measures overlap as: "
             "intersection/union of predicted and ground truth regions. "
             "Range 0-1, stricter than Dice - penalizes both false positives and negatives equally."
    )
    col3.metric(
        "Defect Pixels", f"{int(pred_mask.sum())}",
        help="Total number of pixels classified as defect by the model (after 0.5 threshold). "
             f"Out of {128*128:,} total pixels in the 128x128 image. "
             "Compare with ground truth to assess over/under-segmentation."
    )
    
    st.markdown("---")
    
    # Display images
    st.subheader(
        "Segmentation Results",
        help="Side-by-side comparison of input, ground truth, and model prediction. "
             "Original steel images are 256x1600 pixels, resized to 128x128 for model input. "
             "White pixels indicate defect regions. Compare Ground Truth vs Prediction "
             "to visually assess model accuracy."
    )
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            "**Input Image** ",
            help="Steel surface image from Severstal dataset. Original size 256x1600, "
                 "resized to 128x128 grayscale. The extreme aspect ratio change (6:1 to 1:1) "
                 "may cause some loss of fine defect details."
        )
        st.image(img_input[:, :, 0], clamp=True, use_container_width=True)

    with col2:
        st.markdown(
            "**Ground Truth** ",
            help="Human-annotated defect mask from Kaggle dataset. White pixels show actual "
                 "defect locations. This is the target the model tries to predict. "
                 "Annotations combine all 4 defect classes into binary mask."
        )
        st.image(gt_mask_resized, clamp=True, use_container_width=True)

    with col3:
        st.markdown(
            "**Prediction** ",
            help="U-Net model output after applying 0.5 threshold. Raw model outputs "
                 "continuous values 0-1 (confidence). Pixels above 0.5 are classified as defect. "
                 "Compare with Ground Truth to see hits, misses, and false positives."
        )
        st.image(pred_mask, clamp=True, use_container_width=True)
    
    # Overlay visualization
    st.markdown("---")
    st.subheader(
        "Overlay: Prediction on Original",
        help="Visual overlay showing where the model detected defects on the steel surface. "
             "Red-highlighted areas are pixels the model classified as defects. "
             "This visualization helps assess if predictions align with visible surface anomalies."
    )

    # Create RGB overlay
    overlay = np.stack([img_input[:, :, 0]] * 3, axis=-1)
    overlay[:, :, 0] = np.where(pred_mask > 0, 1.0, overlay[:, :, 0])  # Red channel for defects
    overlay[:, :, 1] = np.where(pred_mask > 0, 0.3, overlay[:, :, 1])  # Dim green
    overlay[:, :, 2] = np.where(pred_mask > 0, 0.3, overlay[:, :, 2])  # Dim blue

    # Display overlay in a centered column to limit width
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(overlay, clamp=True, use_container_width=True)
