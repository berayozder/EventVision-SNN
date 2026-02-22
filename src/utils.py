import numpy as np


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