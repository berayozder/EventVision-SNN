import cv2
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn


class SNNProcessor:
    """
    A V1-cortex-inspired Spiking Neural Network processor.

    Pipeline:
        [B, 2, H, W] spikes
            → Conv2D (8 kernels, 3×3)   # Gabor edge detectors — biologically initialised
            → Leaky LIF neurons          # fires when a feature exceeds threshold
            → [B, 8, H, W] feature spikes

    The Conv2D kernels are initialised as Gabor filters — the same oriented-edge
    detectors found in the primary visual cortex (V1). STDP then refines them
    online based on the patterns seen in the input video.
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
            in_channels=2,           # ON and OFF polarity channels
            out_channels=num_features,
            kernel_size=3,
            padding=1,
            bias=False               # No bias — LIF threshold plays that role
        )

        # Initialise kernels as Gabor filters instead of random weights.
        # This gives the network meaningful edge-detection from frame 1,
        # matching the receptive fields of V1 simple cells in the brain.
        self._init_gabor_kernels(num_features, kernel_size=3)

        # LIF layer: one neuron per (feature, pixel) position.
        # beta defines the 'leak' — how fast the neuron forgets past input.
        self.lif = snn.Leaky(beta=beta, threshold=threshold)

        # Membrane potential state — initialized on first forward pass
        self.mem = None
        self.num_features = num_features

    # ------------------------------------------------------------------
    # Gabor initialisation
    # ------------------------------------------------------------------

    def _init_gabor_kernels(self, num_features: int, kernel_size: int = 3):
        """
        Fill self.conv.weight with Gabor filters at evenly-spaced orientations.

        A Gabor filter is a tiny striped pattern (sine wave inside a Gaussian
        envelope) at a chosen angle. It's the mathematical description of what
        V1 simple cells actually respond to — oriented edges in the visual field.

        We space num_features orientations evenly across 0° → 180°:
            8 filters → 0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°
        """
        kernels = []
        for i in range(num_features):
            # Orientation angle evenly spread from 0 to π (0° to 180°)
            theta = i * np.pi / num_features

            # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi)
            # sigma  — width of the Gaussian envelope  (controls spread)
            # lambda — wavelength of the sine pattern  (controls stripe width)
            # gamma  — aspect ratio                    (1.0 = circular)
            # psi    — phase offset                    (0.0 = cosine phase)
            gabor = cv2.getGaborKernel(
                ksize=(kernel_size, kernel_size),
                sigma=1.5,
                theta=theta,
                lambd=3.0,
                gamma=1.0,
                psi=0.0,
                ktype=cv2.CV_32F,
            )

            # Normalise so all filters have the same scale
            gabor /= (np.abs(gabor).sum() + 1e-8)
            kernels.append(gabor)

        # Stack into tensor: [num_features, 1, kH, kW]
        kernel_tensor = torch.tensor(np.stack(kernels), dtype=torch.float32).unsqueeze(1)

        # Broadcast the same kernel to both input channels (ON and OFF)
        # Result shape: [num_features, 2, kH, kW]
        kernel_tensor = kernel_tensor.expand(-1, 2, -1, -1).contiguous()

        with torch.no_grad():
            self.conv.weight.copy_(kernel_tensor)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

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