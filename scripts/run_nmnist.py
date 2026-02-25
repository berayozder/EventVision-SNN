"""
N-MNIST Classification with STDP + Winner-Take-All Readout
===========================================================

A fully biologically plausible unsupervised classification experiment:

  Phase 1 — STDP Training:
    Feed N-MNIST training samples through the Conv-SNN.
    STDP updates the Gabor-initialized kernels online.
    No labels used.

  Phase 2 — WTA Evaluation:
    Extract mean spike-rate feature vectors for all samples.
    Assign each output neuron to its most-responsive class (WTA).
    Measure classification accuracy on the test set.

Usage:
    python scripts/run_nmnist.py                        # full run
    python scripts/run_nmnist.py --n_train 1000         # quick test
    python scripts/run_nmnist.py --mode train           # train only
    python scripts/run_nmnist.py --mode eval            # eval only
"""

import sys
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import get_nmnist_loader
from processor import SNNProcessor
from stdp import STDPLearner

WEIGHTS_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "trained_weights.pt")
PLOT_PATH     = os.path.join(os.path.dirname(__file__), "..", "data", "kernel_evolution.png")
N_CLASSES = 10


# ---------------------------------------------------------------------------
# Kernel visualisation
# ---------------------------------------------------------------------------

def plot_kernels(before: np.ndarray, after: np.ndarray, save_path: str = PLOT_PATH):
    """
    Plot the 8 Conv2D kernels before and after STDP training side-by-side.

    :param before: numpy array of shape [8, kH, kW] — weights before training
    :param after:  numpy array of shape [8, kH, kW] — weights after training
    :param save_path: file path to save the figure (PNG)
    """
    num_kernels = before.shape[0]

    fig = plt.figure(figsize=(num_kernels * 1.6, 4.2))
    fig.suptitle(
        "Conv2D Kernels: Gabor Init  →  After STDP Training on N-MNIST",
        fontsize=13, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(
        2, num_kernels,
        figure=fig,
        hspace=0.08,
        wspace=0.08,
    )

    row_labels = ["Before (Gabor)", "After (STDP)"]
    rows       = [before, after]

    for row_idx, (label, kernels) in enumerate(zip(row_labels, rows)):
        for col_idx in range(num_kernels):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            k  = kernels[col_idx]          # [kH, kW]

            # Diverging colormap: negative weights = purple, positive = yellow
            vmax = np.abs(k).max() + 1e-8
            ax.imshow(k, cmap="PiYG", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9, labelpad=6)
            if row_idx == 0:
                angle = col_idx * 180 // num_kernels
                ax.set_title(f"{angle}°", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Kernel plot saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Phase 1: STDP Training
# ---------------------------------------------------------------------------

def train(args):
    print(f"\n{'='*55}")
    print(f"  PHASE 1 — STDP Training  ({args.n_train} samples)")
    print(f"{'='*55}")

    processor = SNNProcessor(beta=args.beta, threshold=args.threshold)
    stdp      = STDPLearner(processor.conv, tau=0.9, A_plus=0.005, A_minus=0.005)

    # Snapshot the Gabor-initialized kernels *before* any learning happens.
    # We average the two input channels (ON + OFF) for a clean 2-D view.
    gabor_weights = processor.conv.weight.data.mean(dim=1).cpu().numpy()  # [8, kH, kW]
    loader    = get_nmnist_loader(
        train=True, n_time_bins=args.n_time_bins,
        n_samples=args.n_train, data_root=args.data_root,
    )

    for sample_idx, (frames, label) in enumerate(loader):
        # frames: [1, T, 2, H, W]  — batch size is 1 for online STDP
        frames = frames[0]          # [T, 2, H, W]
        processor.reset()
        stdp.reset()

        # Feed each time bin through the SNN, apply STDP after every frame
        for t in range(frames.shape[0]):
            event_tensor = frames[t].unsqueeze(0)   # [1, 2, H, W]
            spk, _ = processor.process(event_tensor)
            stdp.update(event_tensor, spk)

        if (sample_idx + 1) % 100 == 0:
            print(f"  [{sample_idx+1:5d}/{args.n_train}]  "
                  f"Weight norm: {stdp.weight_norm():.4f}")

    # Save trained weights
    torch.save(processor.conv.weight.data, WEIGHTS_PATH)
    print(f"\n  ✅ Weights saved → {WEIGHTS_PATH}")
    print(f"  Final weight norm: {stdp.weight_norm():.4f}")

    # Snapshot trained kernels and plot before vs after
    trained_weights = processor.conv.weight.data.mean(dim=1).cpu().numpy()  # [8, kH, kW]
    plot_kernels(gabor_weights, trained_weights)


# ---------------------------------------------------------------------------
# Phase 2: Feature Extraction + WTA Readout
# ---------------------------------------------------------------------------

def extract_features(processor, loader, n_samples, n_time_bins):
    """
    For each sample, feed all T frames through the SNN and return
    the mean spike rate per output feature map — a vector of [num_features].
    """
    features, labels_list = [], []

    for sample_idx, (frames, label) in enumerate(loader):
        if sample_idx >= n_samples:
            break
        frames = frames[0]   # [T, 2, H, W]
        processor.reset()

        # Accumulate spike counts over all time bins
        spike_sum = None
        for t in range(frames.shape[0]):
            event_tensor = frames[t].unsqueeze(0)
            spk, _ = processor.process(event_tensor)
            spike_sum = spk if spike_sum is None else spike_sum + spk

        # Mean firing rate per output feature: [num_features]
        mean_rates = spike_sum.mean(dim=[0, 2, 3]) / n_time_bins
        features.append(mean_rates.detach().numpy())
        labels_list.append(label.item())

    return np.array(features), np.array(labels_list)


def wta_classify(train_features, train_labels, test_features, n_classes=10):
    """
    Winner-Take-All readout:
      1. For each neuron, find which class it fires most for → assign it.
      2. For each test sample, the class of the highest-firing neuron is the prediction.
    """
    n_neurons = train_features.shape[1]

    # Step 1: neuron-to-class assignment
    neuron_class = np.zeros(n_neurons, dtype=int)
    for n in range(n_neurons):
        # Mean firing rate of neuron n for each class
        class_rates = np.array([
            train_features[train_labels == c, n].mean()
            if (train_labels == c).any() else 0.0
            for c in range(n_classes)
        ])
        neuron_class[n] = np.argmax(class_rates)

    # Step 2: prediction = class of max-firing neuron
    predictions = np.array([
        neuron_class[np.argmax(feat)] for feat in test_features
    ])
    return predictions


def evaluate(args):
    print(f"\n{'='*55}")
    print(f"  PHASE 2 — WTA Evaluation")
    print(f"{'='*55}")

    if not os.path.exists(WEIGHTS_PATH):
        print("  ❌ No trained weights found. Run with --mode train first.")
        return

    # Load trained weights into a fresh processor
    processor = SNNProcessor(beta=args.beta, threshold=args.threshold)
    processor.conv.weight.data = torch.load(WEIGHTS_PATH)
    print(f"  Loaded weights from {WEIGHTS_PATH}")

    # Extract features for train + test sets
    print(f"  Extracting training features ({args.n_train} samples)...")
    train_loader = get_nmnist_loader(
        train=True, n_time_bins=args.n_time_bins,
        n_samples=args.n_train, data_root=args.data_root,
    )
    train_feat, train_lbl = extract_features(
        processor, train_loader, args.n_train, args.n_time_bins)

    print(f"  Extracting test features    ({args.n_test} samples)...")
    test_loader = get_nmnist_loader(
        train=False, n_time_bins=args.n_time_bins,
        n_samples=args.n_test, data_root=args.data_root,
    )
    test_feat, test_lbl = extract_features(
        processor, test_loader, args.n_test, args.n_time_bins)

    # WTA classification
    predictions = wta_classify(train_feat, train_lbl, test_feat)
    accuracy = (predictions == test_lbl).mean() * 100

    # Per-class accuracy
    print(f"\n  {'Class':<8} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print(f"  {'-'*36}")
    for c in range(N_CLASSES):
        mask = test_lbl == c
        if mask.sum() > 0:
            acc_c = (predictions[mask] == c).mean() * 100
            print(f"  {c:<8} {(predictions[mask]==c).sum():>8} "
                  f"{mask.sum():>8} {acc_c:>7.1f}%")

    print(f"\n  {'='*36}")
    print(f"  Overall WTA Accuracy: {accuracy:.1f}%  "
          f"(chance = {100/N_CLASSES:.0f}%)")
    print(f"  {'='*36}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="N-MNIST unsupervised classification via STDP + WTA"
    )
    parser.add_argument("--mode",        choices=["train", "eval", "both"],
                        default="both",  help="Which phase to run")
    parser.add_argument("--n_train",     type=int, default=5000,
                        help="Number of training samples")
    parser.add_argument("--n_test",      type=int, default=1000,
                        help="Number of test samples")
    parser.add_argument("--n_time_bins", type=int, default=10,
                        help="Time bins per event stream")
    parser.add_argument("--beta",        type=float, default=0.8,
                        help="LIF membrane decay rate")
    parser.add_argument("--threshold",   type=float, default=1.0,
                        help="LIF firing threshold")
    parser.add_argument("--data_root",   type=str, default="./data",
                        help="Directory to store N-MNIST dataset")
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        train(args)
    if args.mode in ("eval", "both"):
        evaluate(args)
