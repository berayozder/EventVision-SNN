"""
MNIST Dataset Verification Script
===================================
Reads the raw Kaggle MNIST IDX files and confirms:
  - Correct number of samples (60 000 train / 10 000 test)
  - Correct image dimensions (28 x 28, uint8, range 0-255)
  - All 10 digit classes present in both splits
  - Saves a 4×8 sample grid image to data/mnist_sample_grid.png

Usage:
    python scripts/verify_mnist.py
    python scripts/verify_mnist.py --data_root ./data/MNIST/raw
"""

import os
import sys
import struct
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# IDX reader (pure stdlib + numpy)
# ---------------------------------------------------------------------------

def read_idx_images(path: str) -> np.ndarray:
    """
    Parse an IDX3-ubyte file and return a uint8 array [N, H, W].

    IDX3 header (big-endian):
        0x00000803  4 bytes  magic
        N           4 bytes  number of images
        H           4 bytes  rows
        W           4 bytes  cols
        data        N*H*W bytes of uint8 pixels
    """
    with open(path, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
    if magic != 0x803:
        raise ValueError(f"Not a valid IDX3 image file: {path!r} (magic={magic:#x})")
    data = np.frombuffer(open(path, "rb").read()[16:], dtype=np.uint8)
    return data.reshape(n, h, w)


def read_idx_labels(path: str) -> np.ndarray:
    """
    Parse an IDX1-ubyte file and return a uint8 array [N].

    IDX1 header (big-endian):
        0x00000801  4 bytes  magic
        N           4 bytes  number of items
        data        N bytes of uint8 labels
    """
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
    if magic != 0x801:
        raise ValueError(f"Not a valid IDX1 label file: {path!r} (magic={magic:#x})")
    data = np.frombuffer(open(path, "rb").read()[8:], dtype=np.uint8)
    return data


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_split(name: str, img_path: str, lbl_path: str, expected_n: int):
    """Load, validate, and print stats for one split (train or test)."""
    print(f"\n{'─'*50}")
    print(f"  [{name}]")
    print(f"{'─'*50}")

    if not os.path.isfile(img_path):
        print(f"  ❌ Image file not found: {img_path}")
        return None, None
    if not os.path.isfile(lbl_path):
        print(f"  ❌ Label file not found: {lbl_path}")
        return None, None

    images = read_idx_images(img_path)
    labels = read_idx_labels(lbl_path)

    # Shape checks
    assert images.shape == (expected_n, 28, 28), \
        f"Unexpected image shape: {images.shape}"
    assert labels.shape == (expected_n,), \
        f"Unexpected label shape: {labels.shape}"

    unique, counts = np.unique(labels, return_counts=True)
    assert len(unique) == 10, f"Expected 10 classes, got {len(unique)}"

    print(f"  Images : {images.shape}  dtype={images.dtype}  "
          f"min={images.min():<3}  max={images.max()}")
    print(f"  Labels : {labels.shape}  classes={sorted(unique.tolist())}")
    print(f"  Label distribution:")
    for cls, cnt in zip(unique, counts):
        bar = "█" * (cnt // 500)
        print(f"    {cls}: {cnt:6d}  {bar}")
    print(f"  ✅ {name} split OK")

    return images, labels


def save_sample_grid(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    out_path: str,
    rows: int = 4,
    cols: int = 8,
):
    """Save a grid of sample MNIST images (one per class shown multiple times)."""
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    fig.suptitle("MNIST Sample Grid  (Kaggle download verification)", fontsize=11)

    rng = np.random.default_rng(seed=0)
    for r in range(rows):
        for c in range(cols):
            cls = (r * cols + c) % 10
            idx_pool = np.where(train_labels == cls)[0]
            idx = rng.choice(idx_pool)
            axes[r, c].imshow(train_images[idx], cmap="gray", interpolation="nearest")
            axes[r, c].set_title(str(cls), fontsize=8, pad=2)
            axes[r, c].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅ Sample grid saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Kaggle MNIST IDX files")
    parser.add_argument(
        "--data_root",
        default=os.path.join(
            os.path.dirname(__file__), "..", "data", "MNIST", "raw"
        ),
        help="Directory containing the 4 MNIST IDX files",
    )
    args = parser.parse_args()
    root = os.path.abspath(args.data_root)

    print(f"\n{'='*50}")
    print(f"  MNIST Verification")
    print(f"  data root: {root}")
    print(f"{'='*50}")

    train_images, train_labels = verify_split(
        "Train Images / Labels",
        img_path=os.path.join(root, "train-images.idx3-ubyte"),
        lbl_path=os.path.join(root, "train-labels.idx1-ubyte"),
        expected_n=60_000,
    )

    test_images, test_labels = verify_split(
        "Test Images / Labels",
        img_path=os.path.join(root, "t10k-images.idx3-ubyte"),
        lbl_path=os.path.join(root, "t10k-labels.idx1-ubyte"),
        expected_n=10_000,
    )

    if train_images is not None and test_images is not None:
        grid_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "mnist_sample_grid.png"
        )
        save_sample_grid(train_images, train_labels, os.path.abspath(grid_path))

        print(f"\n{'='*50}")
        print(f"  ALL CHECKS PASSED ✅")
        print(f"  Dataset is ready for use.")
        print(f"{'='*50}\n")
    else:
        print("\n  ❌ Verification failed. Check the errors above.")
        sys.exit(1)
