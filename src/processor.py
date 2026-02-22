import torch
import torch.nn as nn
import snntorch as snn


class SNNProcessor:
    """
    A V1-cortex-inspired Spiking Neural Network processor.

    Pipeline:
        [B, 2, H, W] spikes
            → Conv2D (8 kernels, 3×3)   # detects oriented edges/shapes
            → Leaky LIF neurons          # fires when a feature exceeds threshold
            → [B, 8, H, W] feature spikes
    """

    def __init__(self, beta=0.8, threshold=1.0, num_features=8):
        """
        :param beta:         Membrane decay rate (0 < beta < 1). Higher = longer memory.
        :param threshold:    Firing threshold for LIF neurons.
        :param num_features: Number of convolutional feature maps (edge detectors).
        """
        # Conv2D: scans the ON/OFF spike map with num_features different 3×3 kernels.
        # padding=1 keeps the spatial size (H, W) unchanged after convolution.
        self.conv = nn.Conv2d(
            in_channels=2,          # ON and OFF polarity channels
            out_channels=num_features,
            kernel_size=3,
            padding=1,
            bias=False              # No bias — LIF threshold plays that role
        )

        # LIF layer: one neuron per (feature, pixel) position.
        # beta defines the 'leak' — how fast the neuron forgets past input.
        self.lif = snn.Leaky(beta=beta, threshold=threshold)

        # Membrane potential state — initialized on first forward pass
        self.mem = None
        self.num_features = num_features

    def process(self, event_tensor):
        """
        Forward pass through Conv2D → LIF.

        :param event_tensor: Float tensor of shape [Batch, 2, H, W]
        :return: spk  — binary spike tensor [Batch, num_features, H, W]
                 mem  — membrane potential   [Batch, num_features, H, W]
        """
        # Step 1: Convolve the spike map — extract oriented edge features
        conv_out = self.conv(event_tensor)   # [B, 8, H, W]

        # Step 2: Initialize membrane state on the first call
        if self.mem is None:
            self.mem = torch.zeros_like(conv_out)

        # Step 3: LIF forward — integrate conv activations into membrane potential
        # spk = 1 wherever membrane exceeded threshold (neuron fired)
        spk, self.mem = self.lif(conv_out, self.mem)

        return spk, self.mem

    def reset(self):
        """Resets the membrane potential state (call when switching video sources)."""
        self.mem = None