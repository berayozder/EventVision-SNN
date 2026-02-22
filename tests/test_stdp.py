"""
Tests for the STDPLearner class.

All tests use synthetic spike tensors — no camera or video required.
"""

import sys
import os
import math
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stdp import STDPLearner
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conv():
    """A small Conv2d layer with weights initialised to 0.5."""
    layer = nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False)
    nn.init.constant_(layer.weight, 0.5)
    return layer


@pytest.fixture
def learner(conv):
    return STDPLearner(conv, tau=0.9, A_plus=0.01, A_minus=0.01)


def make_spikes(shape, density=0.1):
    """Random binary spike tensor with given firing density."""
    return (torch.rand(shape) < density).float()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSTDPLearner:

    def test_weights_change_after_update(self, conv):
        """Weights must change after calling update() with dominant LTP conditions."""
        # Use A_plus >> A_minus so LTP strictly dominates — net change guaranteed
        learner = STDPLearner(conv, A_plus=0.05, A_minus=0.001)
        nn.init.constant_(conv.weight, 0.3)  # start below midpoint
        w_before = conv.weight.data.clone()

        # Dense pre (always active) + sparse post → classic LTP scenario
        pre  = torch.ones(1, 2, 32, 32)
        post = make_spikes((1, 8, 32, 32), density=0.5)
        learner.update(pre, post)
        assert not torch.allclose(conv.weight.data, w_before), \
            "Weights should change after an STDP update with dominant LTP."

    def test_weights_stay_bounded(self, learner, conv):
        """After many updates weights must stay within [w_min, w_max]."""
        pre  = make_spikes((1, 2, 32, 32), density=0.5)
        post = make_spikes((1, 8, 32, 32), density=0.5)
        for _ in range(200):
            learner.update(pre, post)
        assert conv.weight.data.min() >= learner.w_min - 1e-6
        assert conv.weight.data.max() <= learner.w_max + 1e-6

    def test_ltp_dominates_with_correlated_spikes(self, conv):
        """
        When pre fires consistently before post (always active pre, sparse post),
        and A_plus > A_minus, weights should increase on average.
        """
        learner = STDPLearner(conv, A_plus=0.02, A_minus=0.001)
        nn.init.constant_(conv.weight, 0.4)   # start below midpoint

        # Dense pre (always active) + sparse post = pre fires before post → LTP
        pre  = torch.ones(1, 2, 16, 16)
        post = make_spikes((1, 8, 16, 16), density=0.5)
        w_before = conv.weight.data.mean().item()
        for _ in range(50):
            learner.update(pre, post)
        w_after = conv.weight.data.mean().item()
        assert w_after > w_before, \
            "Dominant LTP should increase average weight with correlated spikes."

    def test_ltd_dominates_with_anticorrelated_spikes(self, conv):
        """
        When A_minus > A_plus and post trace is large, weights should decrease.
        """
        learner = STDPLearner(conv, A_plus=0.001, A_minus=0.02)
        nn.init.constant_(conv.weight, 0.6)   # start above midpoint

        post = torch.ones(1, 8, 16, 16)       # post always active → large post_trace
        pre  = make_spikes((1, 2, 16, 16), density=0.5)
        w_before = conv.weight.data.mean().item()
        for _ in range(50):
            learner.update(pre, post)
        w_after = conv.weight.data.mean().item()
        assert w_after < w_before, \
            "Dominant LTD should decrease average weight."

    def test_weight_norm_is_positive(self, learner, conv):
        """weight_norm() should return a positive scalar."""
        norm = learner.weight_norm()
        assert isinstance(norm, float)
        assert norm > 0

    def test_reset_clears_traces(self, learner):
        """After reset, pre_trace and post_trace should be None."""
        pre  = make_spikes((1, 2, 16, 16))
        post = make_spikes((1, 8, 16, 16))
        learner.update(pre, post)
        assert learner.pre_trace is not None
        learner.reset()
        assert learner.pre_trace  is None
        assert learner.post_trace is None

    def test_output_shape_unchanged(self, learner, conv):
        """STDP should not change the shape of the weight tensor."""
        shape_before = tuple(conv.weight.shape)
        pre  = make_spikes((1, 2, 32, 32))
        post = make_spikes((1, 8, 32, 32))
        learner.update(pre, post)
        assert tuple(conv.weight.shape) == shape_before
