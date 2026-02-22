"""
Smoke tests for the EventVision-SNN pipeline.
No camera or hardware required — uses synthetic random frames.
"""

import sys
import os
import numpy as np
import pytest

# Allow imports from src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generator import EventGenerator
from processor import SNNProcessor
from utils import visualize_events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_frame():
    """A random 64x64 BGR frame (uint8), simulating a webcam capture."""
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def generator():
    return EventGenerator(threshold=0.15)


@pytest.fixture
def processor():
    return SNNProcessor(beta=0.8, threshold=1.0)


# ---------------------------------------------------------------------------
# EventGenerator tests
# ---------------------------------------------------------------------------

class TestEventGenerator:

    def test_first_frame_returns_zeros(self, generator, dummy_frame):
        """The very first frame has no reference, so no spikes should fire."""
        on, off = generator.process_frame(dummy_frame)
        assert on.shape == (64, 64)
        assert off.shape == (64, 64)
        assert on.sum() == 0 and off.sum() == 0, "No spikes expected on first frame."

    def test_second_frame_produces_spikes(self, generator, dummy_frame):
        """After the first frame, a different frame should produce some spikes."""
        generator.process_frame(dummy_frame)  # prime the reference
        different_frame = np.zeros((64, 64, 3), dtype=np.uint8)  # black frame → big diff
        on, off = generator.process_frame(different_frame)
        assert on.shape == (64, 64)
        assert off.shape == (64, 64)
        assert off.sum() > 0, "Bright-to-black transition should fire OFF spikes."

    def test_identical_frames_produce_no_spikes(self, generator, dummy_frame):
        """Two identical consecutive frames should produce zero spikes."""
        generator.process_frame(dummy_frame)
        on, off = generator.process_frame(dummy_frame.copy())
        assert on.sum() == 0 and off.sum() == 0

    def test_convert_to_tensor_shape(self, generator, dummy_frame):
        """Tensor output should be [1, 2, H, W]."""
        generator.process_frame(dummy_frame)
        on, off = generator.process_frame(np.zeros_like(dummy_frame))
        tensor = generator.convert_to_tensor(on, off)
        assert tensor.shape == (1, 2, 64, 64)


# ---------------------------------------------------------------------------
# SNNProcessor tests
# ---------------------------------------------------------------------------

class TestSNNProcessor:

    def test_output_shapes(self, generator, processor, dummy_frame):
        """Spike and membrane tensors should match [1, 2, H, W]."""
        generator.process_frame(dummy_frame)
        on, off = generator.process_frame(np.zeros_like(dummy_frame))
        event_tensor = generator.convert_to_tensor(on, off)
        spk, mem = processor.process(event_tensor)
        assert spk.shape == (1, 2, 64, 64)
        assert mem.shape == (1, 2, 64, 64)

    def test_spikes_are_binary(self, generator, processor, dummy_frame):
        """The spike tensor should only contain 0.0 or 1.0 values."""
        generator.process_frame(dummy_frame)
        on, off = generator.process_frame(np.zeros_like(dummy_frame))
        event_tensor = generator.convert_to_tensor(on, off)
        spk, _ = processor.process(event_tensor)
        unique_vals = spk.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique_vals), f"Non-binary spike values: {unique_vals}"


# ---------------------------------------------------------------------------
# Visualize utils tests
# ---------------------------------------------------------------------------

class TestVisualizeEvents:

    def test_output_shape_and_dtype(self, dummy_frame):
        """visualize_events should return uint8 RGB of same H×W."""
        on = np.ones((64, 64), dtype=np.float32)
        off = np.zeros((64, 64), dtype=np.float32)
        vis = visualize_events(on, off)
        assert vis.shape == (64, 64, 3)
        assert vis.dtype == np.uint8

    def test_on_spikes_are_green(self):
        """A pure ON spike should appear in the green channel only."""
        on = np.ones((4, 4), dtype=np.float32)
        off = np.zeros((4, 4), dtype=np.float32)
        vis = visualize_events(on, off)
        assert vis[:, :, 1].max() == 255, "ON spikes should fill green channel."
        assert vis[:, :, 2].max() == 0,   "OFF channel should be empty."

    def test_off_spikes_are_red(self):
        """A pure OFF spike should appear in the red channel only."""
        on = np.zeros((4, 4), dtype=np.float32)
        off = np.ones((4, 4), dtype=np.float32)
        vis = visualize_events(on, off)
        assert vis[:, :, 2].max() == 255, "OFF spikes should fill red channel."
        assert vis[:, :, 1].max() == 0,   "ON channel should be empty."
