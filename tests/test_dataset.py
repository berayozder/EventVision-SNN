import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _tonic_available():
    try:
        import tonic
        return True
    except ImportError:
        return False


def _dataset_available():
    """Try to import and initialise the dataset — skip if download fails."""
    try:
        from dataset import get_nmnist_loader
        loader = get_nmnist_loader(train=True, n_time_bins=10,
                                   n_samples=2, data_root="./data")
        next(iter(loader))   # trigger actual download
        return True
    except Exception:
        return False

@pytest.mark.skipif(
    not _tonic_available() or not _dataset_available(),
    reason="tonic not installed or N-MNIST unavailable (network/SSL)"
)
class TestNMNISTLoader:

    @pytest.fixture(scope="class")
    def loader(self):
        """Load a tiny slice (10 samples) of N-MNIST training set."""
        from dataset import get_nmnist_loader
        return get_nmnist_loader(
            train=True, n_time_bins=10, n_samples=10, data_root="./data"
        )

    def test_batch_shape(self, loader):
        """Each batch should be [B, T, 2, H, W] = [1, 10, 2, 34, 34]."""
        frames, labels = next(iter(loader))
        assert frames.ndim == 5, f"Expected 5-D tensor, got {frames.ndim}-D"
        B, T, C, H, W = frames.shape
        assert T  == 10, f"Expected 10 time bins, got {T}"
        assert C  == 2,  f"Expected 2 polarity channels, got {C}"
        assert H  == 34, f"Expected H=34, got {H}"
        assert W  == 34, f"Expected W=34, got {W}"

    def test_labels_in_range(self, loader):
        """Labels must be integers in [0, 9]."""
        for _, labels in loader:
            assert labels.dtype == torch.long
            assert labels.min() >= 0
            assert labels.max() <= 9

    def test_frames_are_non_negative(self, loader):
        """Spike counts must be >= 0."""
        frames, _ = next(iter(loader))
        assert frames.min() >= 0, "Spike frame values should be non-negative."

    def test_loader_length(self, loader):
        """Loader should have exactly n_samples batches (batch_size=1)."""
        assert len(loader) == 10
