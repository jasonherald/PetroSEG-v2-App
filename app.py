import streamlit as st
import cv2
import numpy as np
import os
from skimage import segmentation
from skimage.color import color_dict, label2rgb
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
    # Generate a unique color for each segment
    unique_labels = np.unique(segmented_image.reshape(-1, 3), axis=0)
    new_colors = np.random.randint(0, 256, (len(unique_labels), 3), dtype=np.uint8)
    
    # Apply the new colors to the segmented image
    for i, label in enumerate(unique_labels):
        mask = np.all(segmented_image == label, axis=-1)
        segmented_image[mask] = new_colors[i]
    
    return segmented_image

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
        image_flatten = image.reshape((-1, 3))
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

        image_flatten = image.reshape((-1, 3))
        color_avg = np.random.randint(255, size=(args.max_label_num, 3))
        show = image

        progress_bar = st.progress(0)
        image_placeholder = st.empty()

        for batch_idx in range(args.train_epoch):
            with tf.GradientTape() as tape:
                # Forward pass
                output = model(tensor, training=True)[0]
                output = tf.reshape(output, (-1, args.mod_dim2))
                target = tf.argmax(output, axis=1)
                im_target = target.numpy()

                # Update target labels based on segmentation
                for inds in seg_lab:
                    u_labels, hist = np.unique(im_target[inds], return_counts=True)
                    im_target[inds] = u_labels[np.argmax(hist)]

                target = tf.convert_to_tensor(im_target)
                loss = criterion(target, output)

            # Backward pass and optimization
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables)) # type: ignore

            # Update segmented image
            un_label, lab_inverse = np.unique(im_target, return_inverse=True)
            if un_label.shape[0] < args.max_label_num:
                img_flatten = image_flatten.copy()
                if len(color_avg) != un_label.shape[0]:
                    color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=int) for label in un_label]
                for lab_id, color in enumerate(color_avg):
                    img_flatten[lab_inverse == lab_id] = color
                show = img_flatten.reshape(image.shape)

            segmented_images.append(show.copy())

            # Update progress bar and display image
            progress = (batch_idx + 1) / args.train_epoch
            progress_bar.progress(progress)
            image_placeholder.image(show, caption=f'Epoch {batch_idx + 1}', use_container_width=True)

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
    """Randomize colors for segmentation labels"""
    unique_labels = np.unique(st.session_state.segmented_image.reshape(-1, 3), axis=0)
    random_colors = {tuple(label): tuple(np.random.randint(0, 256, size=3)) for label in unique_labels}

    for old_color, new_color in random_colors.items():
        mask = np.all(st.session_state.segmented_image == np.array(old_color), axis=-1)
        st.session_state.segmented_image[mask] = new_color

    st.session_state.new_colors.update(random_colors)
    st.session_state.image_update_trigger += 1  # Trigger image update

def handle_color_picking() -> None:
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
    """Calculate and display label percentages"""
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

