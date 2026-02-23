"""
N-MNIST Dataset Loader
======================
N-MNIST is the standard MNIST handwritten digit dataset recorded with a
Dynamic Vision Sensor (DVS) camera. Instead of static images, each sample
is a stream of asynchronous events (timestamp, x, y, polarity).

We use the `tonic` library (the community standard for neuromorphic datasets)
to download and convert the events into time-binned spike frames that our
existing Conv-SNN pipeline can process.

Output shape per sample: [T, 2, H, W]
  T = number of time bins (default 10)
  2 = ON and OFF polarity channels
  H, W = 34 x 34 (N-MNIST sensor resolution)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import tonic
import tonic.transforms as transforms


# N-MNIST sensor resolution (fixed by the hardware used to record it)
SENSOR_SIZE = tonic.datasets.NMNIST.sensor_size   # (34, 34, 2)


def get_nmnist_loader(
    train: bool = True,
    batch_size: int = 1,
    n_time_bins: int = 10,
    n_samples: int = None,
    data_root: str = "./data",
) -> DataLoader:
    """
    Returns a DataLoader for the N-MNIST dataset.

    On first call this downloads N-MNIST (~180 MB) into data_root automatically.

    :param train:       True for training split, False for test split.
    :param batch_size:  Samples per batch (keep at 1 for STDP online learning).
    :param n_time_bins: Number of time windows to divide each event stream into.
    :param n_samples:   If set, only use the first n_samples (useful for quick runs).
    :param data_root:   Directory to download / cache the dataset.
    :return:            PyTorch DataLoader yielding (spike_frames, label) pairs.
    """
    # ToFrame accumulates events into fixed time bins.
    # Each bin becomes one "frame" of shape [2, H, W] (ON/OFF channels).
    frame_transform = transforms.Compose([
        transforms.ToFrame(
            sensor_size=SENSOR_SIZE,
            n_time_bins=n_time_bins,
        )
    ])

    dataset = tonic.datasets.NMNIST(
        save_to=data_root,
        train=train,
        transform=frame_transform,
    )

    if n_samples is not None:
        dataset = Subset(dataset, range(min(n_samples, len(dataset))))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=_collate,   # handles variable-length event streams
    )


def _collate(batch):
    """
    Custom collate: stack spike frames and labels into tensors.
    tonic returns numpy arrays; we convert to float32 torch tensors here.
    """
    frames, labels = zip(*batch)
    frames_t = torch.stack([torch.tensor(f, dtype=torch.float32) for f in frames])
    labels_t = torch.tensor(labels, dtype=torch.long)
    return frames_t, labels_t
