import numpy as np
import cv2


def visualize_events(on_spikes: np.ndarray, off_spikes: np.ndarray) -> np.ndarray:
    """
    Creates an RGB visualization of ON and OFF spike maps.

    - ON  spikes → green channel
    - OFF spikes → red channel
    - Overlap (both ON and OFF at same pixel) → yellow

    :param on_spikes:  2-D float32 array (H, W), values in {0, 1}
    :param off_spikes: 2-D float32 array (H, W), values in {0, 1}
    :returns: uint8 RGB image of shape (H, W, 3)
    """
    h, w = on_spikes.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Red channel  → OFF events
    canvas[:, :, 2] = (off_spikes * 255).astype(np.uint8)
    # Green channel → ON events
    canvas[:, :, 1] = (on_spikes * 255).astype(np.uint8)

    return canvas


def visualize_feature_maps(mem, cols: int = 4) -> np.ndarray:
    """
    Tiles the Conv-SNN membrane potential maps into a single grid image.

    Each cell in the grid shows one feature map (edge detector) as a
    normalized grayscale patch, making it easy to see which orientations
    are responding to motion in the scene.

    :param mem:  PyTorch tensor of shape [Batch, num_features, H, W]
    :param cols: Number of columns in the tile grid (rows inferred automatically)
    :returns:    uint8 grayscale image — the full tiled grid
    """
    # Extract all feature maps from the first batch item: [num_features, H, W]
    maps = mem[0].detach().numpy()
    num_features, h, w = maps.shape

    rows = (num_features + cols - 1) // cols   # ceil division
    grid = np.zeros((rows * h, cols * w), dtype=np.float32)

    for idx, fmap in enumerate(maps):
        r, c = divmod(idx, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = fmap

    # Normalize to [0, 255] for display
    grid_min, grid_max = grid.min(), grid.max()
    if grid_max > grid_min:
        grid = (grid - grid_min) / (grid_max - grid_min)

    return (grid * 255).astype(np.uint8)