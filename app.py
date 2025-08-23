import streamlit as st
import cv2
import numpy as np
import os
from skimage import segmentation
from skimage.color import label2rgb
import tensorflow as tf
from keras import layers, models, optimizers, losses
from sklearn.cluster import KMeans

import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries


def resize_image(image, target_size):
    """Resize image while maintaining aspect ratio."""
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate the aspect ratio of the original image
    aspect_ratio = original_width / original_height

    # Determine the new dimensions while maintaining the aspect ratio
    if target_width / target_height > aspect_ratio:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image

def automatically_change_segment_colors(segmented_image):
    """Vectorized recoloring: assign a random color to each unique RGB triplet."""
    view = segmented_image.reshape(-1, 3)
    uniq, inv = np.unique(view, axis=0, return_inverse=True)
    new_colors = np.random.randint(0, 256, size=uniq.shape, dtype=np.uint8)
    remapped = new_colors[inv].reshape(segmented_image.shape)
    return remapped

def download_image(image_array, file_name):
    """Use macOS 'choose file name' to get a save path and write the image there (no Streamlit modal).
    Falls back to ~/Downloads/PetroSeg if AppleScript is unavailable/blocked."""
    try:
        # --- Normalize image dtype to uint8 ---
        img = image_array
        if img.dtype != np.uint8:
            maxv = float(img.max()) if img.size else 1.0
            if maxv <= 1.0:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        # --- Ensure channel order for OpenCV writers ---
        if img.ndim == 3:
            if img.shape[2] == 3:
                # RGB -> BGR
                img_to_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.shape[2] == 4:
                # RGBA -> BGRA (preserve alpha if saving PNG/TIFF)
                img_to_write = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            else:
                img_to_write = img
        else:
            # Grayscale 2D
            img_to_write = img

        # --- Ask for save path using macOS AppleScript ---
        safe_default = file_name.replace('"', '\\"')  # escape quotes for AppleScript string
        script_lines = [
            f'set defaultName to "{safe_default}"',
            'set fp to POSIX path of (choose file name with prompt "Save segmented image as…" default name defaultName)',
            'return fp'
        ]

        save_path = None
        try:
            proc = subprocess.run([
                "osascript", "-e", script_lines[0], "-e", script_lines[1], "-e", script_lines[2]
            ], capture_output=True, text=True)
            if proc.returncode == 0:
                out = proc.stdout.strip()
                if out:
                    save_path = out
        except FileNotFoundError:
            # osascript not present (very rare on macOS), fall through to fallback
            pass

        if not save_path:
            # --- Hardened runtime / automation disabled / user canceled -> fallback path ---
            # If user canceled (non-zero return with some output), just notify and return.
            if proc.returncode != 0 if 'proc' in locals() else False:
                st.info('Save canceled.')
                return

            fallback_dir = Path.home() / 'Downloads' / 'PetroSeg'
            fallback_dir.mkdir(parents=True, exist_ok=True)
            # Ensure extension
            base = Path(file_name)
            # Always use .png extension for fallback
            base = base.with_suffix('.png')
            # Avoid collisions
            candidate = fallback_dir / base.name
            idx = 1
            while candidate.exists():
                stem = base.stem
                suffix = base.suffix
                candidate = fallback_dir / f"{stem}_{idx}{suffix}"
                idx += 1
            save_path = str(candidate)

        # Always save as .png regardless of what the user selected
        p = Path(save_path).with_suffix('.png')

        # --- Write always as PNG ---
        success = cv2.imwrite(str(p), img_to_write)
        if not success:
            st.error('Could not save image.')
            return

        # --- Notify ---
        st.success(f'Saved image to: {str(p)}')
    except Exception as e:
        st.error(f'An error occurred while saving: {e}')

 # --- Vectorized helpers ---
def _vectorized_mean_colors(lab_inverse: np.ndarray, image_flatten: np.ndarray, k: int) -> np.ndarray:
    """Compute per-cluster mean RGB colors using vectorized bincounts.
    Returns uint8 array of shape (k, 3).
    """
    # counts per label (avoid divide-by-zero by replacing zeros with 1)
    counts = np.bincount(lab_inverse, minlength=k).astype(np.float32)
    counts[counts == 0] = 1.0
    # sums for each channel
    sums_r = np.bincount(lab_inverse, weights=image_flatten[:, 0], minlength=k)
    sums_g = np.bincount(lab_inverse, weights=image_flatten[:, 1], minlength=k)
    sums_b = np.bincount(lab_inverse, weights=image_flatten[:, 2], minlength=k)
    color_avg = np.stack([sums_r, sums_g, sums_b], axis=1) / counts[:, None]
    return np.clip(color_avg, 0, 255).astype(np.uint8)

def perform_custom_segmentation(image, params):
    class Args(object):
        def __init__(self, params):
            self.train_epoch = params.get('train_epoch', 2 ** 3)
            self.mod_dim1 = params.get('mod_dim1', 64)
            self.mod_dim2 = params.get('mod_dim2', 32)
            self.gpu_id = params.get('gpu_id', 0)
            self.min_label_num = params.get('min_label_num', 6)
            self.max_label_num = params.get('max_label_num', 256)
            self.segmentation_method = params.get('segmentation_method', 'felzenszwalb')

    args = Args(params)

    def MyNet(inp_dim, mod_dim1, mod_dim2, seed=42):
        tf.random.set_seed(seed)
        
        inputs = layers.Input(shape=(None, None, inp_dim))
        
        # First convolutional block
        x = layers.Conv2D(mod_dim1, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        skip1 = x  # Skip connection 1
        
        # Second convolutional block
        x = layers.Conv2D(mod_dim2, (1, 1))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Third convolutional block
        x = layers.Conv2D(mod_dim1, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        skip2 = x  # Skip connection 2
        
        # Fourth convolutional block
        x = layers.Conv2D(mod_dim2, (1, 1))(x)
        x = layers.BatchNormalization()(x)
        
        # Adding skip connection 2
        skip2 = layers.Conv2D(mod_dim2, (1, 1))(skip2)
        skip2 = layers.BatchNormalization()(skip2)
        
        x = layers.Add()([x, skip2])
        x = layers.ReLU()(x)
        
        # Adding skip connection 1
        skip1 = layers.Conv2D(mod_dim2, (1, 1))(skip1)
        skip1 = layers.BatchNormalization()(skip1)
        
        x = layers.Add()([x, skip1])
        x = layers.ReLU()(x)
        
        model = models.Model(inputs=inputs, outputs=x)
        return model


    np.random.seed(1943)
    tf.random.set_seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Precompute shapes and flattened image as float32 for bincount-based means
    h, w = image.shape[:2]
    image_flatten = image.reshape((-1, 3)).astype(np.float32)

    # Store per-epoch label maps so that downstream stats use exact labels
    label_maps: list[np.ndarray] = []

    '''segmentation ML'''
    if args.segmentation_method == 'felzenszwalb':
        # Perform Felzenszwalb segmentation
        seg_map = segmentation.felzenszwalb(image, scale=15, sigma=0.06, min_size=14)
        seg_map = seg_map.flatten()
        seg_lab = [np.where(seg_map == u_label)[0]
                for u_label in np.unique(seg_map)]

        # Convert segmentation map to RGB image with boundaries
        segmented_image = label2rgb(seg_map.reshape(image.shape[:2]), image, kind='avg')
        boundaries = find_boundaries(seg_map.reshape(image.shape[:2]), mode='thick')
        segmented_image[boundaries] = [1, 0, 0]  # Red color for boundaries
        segmented_image = (segmented_image - segmented_image.min()) / (segmented_image.max() - segmented_image.min())
        st.image(segmented_image, caption='Felzenszwalb Segmentation with Contours', use_container_width=True)
        
    elif args.segmentation_method == 'kmeans':
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=args.max_label_num, random_state=0).fit(image_flatten)
        seg_map = kmeans.labels_
        seg_lab = [np.where(seg_map == u_label)[0] for u_label in np.unique(seg_map)]

    # Set device to GPU if available, otherwise CPU
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    tensor = image.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)

    segmented_images = []

    with tf.device(device):
        # Initialize the model
        model = MyNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2)
        criterion = losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = optimizers.SGD(learning_rate=5e-2, momentum=0.9)

        # Compiled forward pass without XLA (jit_compile) to avoid platform issues on some mac builds
        def forward(t: tf.Tensor):
            x = model(t, training=True)[0]
            return tf.reshape(x, (-1, args.mod_dim2))
        forward = tf.function(forward, reduce_retracing=True)

        color_avg = np.random.randint(255, size=(args.max_label_num, 3))
        show = image

        progress_bar = st.progress(0)
        image_placeholder = st.empty()

        for batch_idx in range(args.train_epoch):
            with tf.GradientTape() as tape:
                # Forward pass
                output = forward(tf.convert_to_tensor(tensor))
                target = tf.argmax(output, axis=1)
                im_target = target.numpy()

                # Update target labels based on segmentation
                for inds in seg_lab:
                    u_labels, hist = np.unique(im_target[inds], return_counts=True)
                    im_target[inds] = u_labels[np.argmax(hist)]

                label_maps.append(im_target.reshape(h, w))

                target = tf.cast(tf.convert_to_tensor(im_target), tf.int32)
                loss = criterion(target, output)

            # Backward pass and optimization
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables)) # type: ignore

            # Update segmented image (vectorized color assignment)
            un_label, lab_inverse = np.unique(im_target, return_inverse=True)
            K = un_label.shape[0]
            if K < args.max_label_num:
                # Compute/refresh mean colors per label using bincounts (fast & deterministic)
                color_avg = _vectorized_mean_colors(lab_inverse, image_flatten, K)
                # Map each pixel to its cluster mean color without per-label loops
                show = color_avg[lab_inverse].reshape(h, w, 3)
            else:
                show = image  # fallback, though unlikely with typical settings

            segmented_images.append(show.copy())

            # Update progress bar and display image
            progress = (batch_idx + 1) / args.train_epoch
            progress_bar.progress(progress)
            image_placeholder.image(show, caption=f'Epoch {batch_idx + 1}', use_container_width=True)

    # Expose exact per-epoch label maps for downstream percentage calculations
    st.session_state["segmented_label_maps"] = label_maps

    return segmented_images

# Constants - this should depend on available RAM
TARGET_SIZE = (750, 750)

def get_parameters_from_sidebar() -> dict:
    """Get segmentation parameters from sidebar"""
    st.sidebar.header("Segmentation Parameters")
    param_names = ['train_epoch', 'mod_dim1', 'mod_dim2', 'min_label_num', 'max_label_num']
    param_values = [(1, 200, 43), (1, 128, 67), (1, 128, 63), (1, 20, 3), (1, 200, 25)]
    params = {name: st.sidebar.slider(name.replace('_', ' ').title(), *values) for name, values in zip(param_names, param_values)}
    
    # Add sliders for target size width and height
    target_size_width = st.sidebar.number_input("Target Size Width", 100, 1200, 750)
    target_size_height = st.sidebar.number_input("Target Size Height", 100, 1200, 750)
    params['target_size'] = (target_size_width, target_size_height) # type: ignore
    
    # Add dropdown for segmentation method
    params['segmentation_method'] = st.sidebar.selectbox('Segmentation Method', ['felzenszwalb', 'kmeans'], index=0) # type: ignore
    
    return params

def display_segmentation_results() -> None:
    """Display segmentation results"""
    st.image(st.session_state.segmented_image, caption='Updated Segmented Image', use_container_width=True)

def randomize_colors() -> None:
    """Randomize colors for segmentation labels using a vectorized LUT remap."""
    img = st.session_state.segmented_image
    # Build a view that makes each RGB triplet a single item for uniqueness
    view = img.reshape(-1, 3)
    uniq, inv = np.unique(view, axis=0, return_inverse=True)
    new_colors = np.random.randint(0, 256, size=uniq.shape, dtype=np.uint8)
    remapped = new_colors[inv].reshape(img.shape)
    st.session_state.segmented_image = remapped

    # Track mapping for color picker compatibility
    st.session_state.new_colors.update({tuple(old): tuple(new) for old, new in zip(map(tuple, uniq), map(tuple, new_colors))})
    st.session_state.image_update_trigger += 1

def handle_color_picking() -> None:
    if st.session_state.segmented_image is None:
        return
    """Handle color picking and other functionalities"""
    unique_labels = np.unique(st.session_state.segmented_image.reshape(-1, 3), axis=0)
    for i, label in enumerate(unique_labels):
        hex_label = f'#{label[0]:02x}{label[1]:02x}{label[2]:02x}'
        new_color = st.color_picker(f"Choose a new color for label {i}", value=hex_label, key=f"label_{i}")
        new_color_rgb = tuple(int(new_color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
        st.session_state.new_colors[tuple(label)] = new_color_rgb

    new_colors_hex = {tuple(label): f'#{label[0]:02x}{label[1]:02x}{label[2]:02x}' for label in st.session_state.new_colors.values()}

    for old_color, new_color in st.session_state.new_colors.items():
        old_color_hex = f'#{old_color[0]:02x}{old_color[1]:02x}{old_color[2]:02x}'
        new_color_hex = new_colors_hex[new_color]
        mask = np.all(st.session_state.segmented_image == np.array(old_color), axis=-1)
        st.session_state.segmented_image[mask] = new_color

    st.session_state.image_update_trigger += 1

def calculate_and_display_label_percentages() -> None:
    """Calculate and display label percentages based on the exact label map for the selected epoch.
    Falls back to RGB-based approximation only if label map is unavailable.
    """
    label_map = st.session_state.get("current_label_map")
    seg_img = st.session_state.get("segmented_image")

    if label_map is not None and seg_img is not None:
        # Exact counts from integer label map
        unique_labels, counts = np.unique(label_map, return_counts=True)
        total_pixels = int(counts.sum()) if counts.size else 1
        label_percentages = {int(lbl): (float(cnt) / float(total_pixels)) * 100.0 for lbl, cnt in zip(unique_labels, counts)}
        # Vectorized: take first occurrence of each label in flattened arrays
        labels_flat = label_map.ravel()
        seg_flat = seg_img.reshape(-1, seg_img.shape[-1])
        _, first_idx = np.unique(labels_flat, return_index=True)
        # Build color mapping
        label_to_color = {}
        for lbl, idx in zip(np.unique(labels_flat), first_idx):
            color = seg_flat[idx]
            hex_color = f'#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}'
            label_to_color[int(lbl)] = hex_color

        st.write("Label Percentages:")
        for lbl in unique_labels:
            hex_color = label_to_color[int(lbl)]
            pct = label_percentages[int(lbl)]
            color_box = (
                f'<div style="display:inline-block;width:20px;height:20px;'
                f'background-color:{hex_color};margin-right:10px;"></div>'
            )
            st.markdown(f'{color_box} Label {int(lbl)}: {pct:.2f}%', unsafe_allow_html=True)
        return

    # Fallback (approximate) — use grayscale buckets only if label map is missing
    final_labels = cv2.cvtColor(st.session_state.segmented_image, cv2.COLOR_BGR2GRAY)
    unique_labels, counts = np.unique(final_labels, return_counts=True)
    total_pixels = np.sum(counts)
    label_percentages = {int(label): (count / total_pixels) * 100 for label, count in zip(unique_labels, counts)}

    label_to_color = {}
    for label in unique_labels:
        mask = final_labels == label
        corresponding_color = st.session_state.segmented_image[mask][0]
        hex_color = f'#{corresponding_color[0]:02x}{corresponding_color[1]:02x}{corresponding_color[2]:02x}'
        label_to_color[int(label)] = hex_color

    st.write("Label Percentages:")
    for label, percentage in label_percentages.items():
        hex_color = label_to_color[label]
        color_box = f'<div style="display: inline-block; width: 20px; height: 20px; background-color: {hex_color}; margin-right: 10px;"></div>'
        st.markdown(f'{color_box} Label {label}: {percentage:.2f}%', unsafe_allow_html=True)

def main() -> None:
    st.title("PetroSeg")
    st.info("""
    - **Training Epochs**: Higher values will lead to fewer segments but may take more time.
    - **Image Size**: For better efficiency, upload small-sized images.
    - **Cache**: For best results, clear the cache between different image uploads. You can do this from the menu in the top-right corner.
    """)

    st.markdown("""
        <style>
            .reportview-container {
                margin-top: -2em;
            }
            
            .stAppDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
        </style>
    """, unsafe_allow_html=True)

    # if torch.cuda.is_available():
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    if 'segmented_images' not in st.session_state:
        st.session_state.segmented_images = []
    if 'new_colors' not in st.session_state:
        st.session_state.new_colors = {}
    if 'image_update_trigger' not in st.session_state:
        st.session_state.image_update_trigger = 0

    params = get_parameters_from_sidebar()

    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "bmp", "tiff", "webp"])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if image is None:
            st.error("Error loading image. Please check the file and try again.")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption='Original Image', use_container_width=True)

        target_size = params['target_size']
        image_resized = resize_image(image_rgb, target_size)

        if st.sidebar.button("Start Segmentation"):
            st.session_state.segmented_images = perform_custom_segmentation(image_resized, params)

        if st.sidebar.button("Change Colors"):
            randomize_colors()
        if st.session_state.segmented_images:
            epoch = st.sidebar.slider("Select Epoch", 1, len(st.session_state.segmented_images), len(st.session_state.segmented_images))
            st.session_state.segmented_image = st.session_state.segmented_images[epoch - 1]
            # Keep the exact integer label map aligned with the selected epoch (for accurate stats)
            seg_label_maps = st.session_state.get("segmented_label_maps")
            if isinstance(seg_label_maps, list) and len(seg_label_maps) >= epoch:
                st.session_state["current_label_map"] = seg_label_maps[epoch - 1]
            handle_color_picking()
            display_segmentation_results()
            calculate_and_display_label_percentages()
            if st.sidebar.button('Save Image…'):
                download_image(st.session_state.segmented_image, 'segmented_image.png')

def initialize_session_state():
    st.session_state.setdefault('segmented_image', None)
    st.session_state.setdefault('new_colors', {})  

if __name__ == "__main__":
    initialize_session_state()
    main()

