import numpy as np
import cv2
from utils import preprocess_canvas_image, center_by_center_of_mass

def test_preprocessing():
    print("Testing utils.py preprocessing pipeline...")

    # 1. Test blank canvas
    blank_canvas = np.zeros((280, 280, 4), dtype=np.uint8)
    blank_canvas[:, :, 3] = 255 # opaque black canvas
    img, is_blank = preprocess_canvas_image(blank_canvas)
    assert is_blank, "Blank canvas should be detected as blank"
    assert img.shape == (2828,) if img.ndim==1 else (28, 28), "Output shape should be (28, 28)"
    print("[OK] Blank canvas test passed!")

    # 2. Test mock drawing of a line/stroke (representing digit '1')
    mock_canvas = np.zeros((280, 280, 4), dtype=np.uint8)
    mock_canvas[:, :, 3] = 255 # background black
    # Draw vertical white stroke at offset location (x=50..70, y=100..220)
    mock_canvas[100:220, 50:70, 0:3] = 255 # white stroke
    
    proc_img, is_blank = preprocess_canvas_image(mock_canvas)
    assert not is_blank, "Non-blank canvas should not be flagged as blank"
    assert proc_img.shape == (28, 28), "Output shape must be (28, 28)"
    assert proc_img.max() > 0.5, "Stroke intensity should be normalized near 1.0"
    assert 0.0 <= proc_img.min() <= 1.0, "Pixel values must be in [0, 1] range"
    print("[OK] Drawing preprocessing & normalization test passed!")

    # 3. Test Center of Mass alignment
    centered = center_by_center_of_mass((proc_img * 255).astype(np.uint8))
    assert centered.shape == (28, 28), "Centered image shape must be (28, 28)"
    print("[OK] Center-of-mass centering test passed!")

    print("ALL UTILS TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_preprocessing()
