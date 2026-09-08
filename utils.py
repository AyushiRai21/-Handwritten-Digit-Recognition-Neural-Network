import numpy as np
from PIL import Image, ImageOps
import cv2

def preprocess_canvas_image(image_data):
    """
    Preprocess user drawing from canvas to match MNIST dataset distribution:
    1. Grayscale conversion & threshold/inversion (digit white, background black)
    2. Bounding box cropping with aspect ratio preservation
    3. Resizing to ~20x20 inside a 28x28 canvas
    4. Center of mass centering
    5. Normalization to [0.0, 1.0]
    
    Returns:
        processed_img: np.ndarray of shape (28, 28) normalized to [0, 1]
        is_blank: bool indicating if canvas had no drawing
    """
    if image_data is None:
        return np.zeros((28, 28), dtype=np.float32), True

    # Convert PIL Image or numpy array to uint8 numpy array
    if isinstance(image_data, Image.Image):
        img_np = np.array(image_data)
    else:
        img_np = np.array(image_data, dtype=np.uint8)

    # Handle RGBA / RGB / Grayscale
    if img_np.ndim == 3:
        if img_np.shape[2] == 4:
            # Check if alpha channel carries the stroke info
            alpha = img_np[:, :, 3]
            rgb = img_np[:, :, :3]
            # Convert RGB to grayscale
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            # If alpha is active where drawing occurs
            if np.max(alpha) > 0 and np.std(alpha) > 5:
                # Use stroke intensity weighted by alpha
                gray = cv2.bitwise_and(gray, gray, mask=alpha)
        else:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    elif img_np.ndim == 2:
        gray = img_np.copy()
    else:
        return np.zeros((28, 28), dtype=np.float32), True

    # Determine background color (white vs black background)
    # MNIST digits are WHITE strokes (255) on BLACK background (0)
    mean_val = np.mean(gray)
    if mean_val > 127:
        # Background is light/white, invert it
        gray = 255 - gray

    # Check for blank canvas
    if np.max(gray) < 25:
        return np.zeros((28, 28), dtype=np.float32), True

    # Find bounding box of foreground drawing
    coords = cv2.findNonZero(gray)
    if coords is None:
        return np.zeros((28, 28), dtype=np.float32), True

    x, y, w, h = cv2.boundingRect(coords)

    # Crop digit
    cropped = gray[y:y+h, x:x+w]

    # Create square canvas for cropped region preserving aspect ratio
    max_dim = max(w, h)
    # Add padding proportional to size
    pad = int(max_dim * 0.15)
    padded_dim = max_dim + 2 * pad

    square_crop = np.zeros((padded_dim, padded_dim), dtype=np.uint8)
    
    # Calculate offset to center cropped digit inside square canvas
    y_offset = (padded_dim - h) // 2
    x_offset = (padded_dim - w) // 2
    square_crop[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    # Resize to 20x20 pixels (standard MNIST inner digit size)
    resized_digit = cv2.resize(square_crop, (20, 20), interpolation=cv2.INTER_AREA)

    # Embed in 28x28 canvas (4-pixel border around 20x20)
    canvas_28 = np.zeros((28, 28), dtype=np.uint8)
    canvas_28[4:24, 4:24] = resized_digit

    # Shift image by center of mass (MNIST normalization standard)
    canvas_28 = center_by_center_of_mass(canvas_28)

    # Normalize pixel values to [0.0, 1.0]
    normalized_img = canvas_28.astype(np.float32) / 255.0

    return normalized_img, False

def center_by_center_of_mass(img_28):
    """
    Shifts a 28x28 grayscale image so its center of mass rests at (14, 14).
    """
    moments = cv2.moments(img_28)
    if moments['m00'] == 0:
        return img_28

    # Calculate center of mass coordinates
    cy = moments['m10'] / moments['m00']
    cx = moments['m01'] / moments['m00']

    # Compute required shift to bring center of mass to (14, 14)
    shift_x = np.round(14.0 - cx).astype(int)
    shift_y = np.round(14.0 - cy).astype(int)

    # Limit maximum shift to prevent clipping
    shift_x = np.clip(shift_x, -4, 4)
    shift_y = np.clip(shift_y, -4, 4)

    # Apply transformation matrix
    M = np.float32([[1, 0, shift_y], [0, 1, shift_x]])
    centered_img = cv2.warpAffine(img_28, M, (28, 28), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return centered_img
