"""
STDP (Spike-Timing Dependent Plasticity) learning rule for Conv2D layers.

Biological basis:
  "Neurons that fire together, wire together."
  — pre fires BEFORE post → strengthen (LTP)
  — pre fires AFTER  post → weaken   (LTD)

Implementation uses synaptic *traces* (exponential moving averages of spike
activity) rather than exact spike timing, which is the standard approximation
in the neuromorphic literature (Morrison et al., 2008).

Weight update rule:
    x_pre  ← τ · x_pre  + pre_spikes          (pre-synaptic trace)
    x_post ← τ · x_post + post_spikes         (post-synaptic trace)

    ΔW = A_+ · corr(x_pre,    post_spikes)    (LTP)
       - A_- · corr(pre_spikes, x_post)       (LTD)

    W ← clamp(W + ΔW, w_min, w_max)           (bounded Hebbian)

The correlation for a Conv2D layer is computed via F.unfold + einsum,
which mirrors the structure of the conv backward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STDPLearner:
    """
    Online STDP learning rule for a single nn.Conv2d layer.

    Usage:
        stdp = STDPLearner(conv_layer)
        # inside your frame loop:
        stdp.update(pre_spikes, post_spikes)
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        tau: float = 0.9,       # trace decay constant (higher = longer memory)
        A_plus: float = 0.005,  # LTP learning rate
        A_minus: float = 0.005, # LTD learning rate
        w_min: float = 0.0,     # lower weight bound
        w_max: float = 1.0,     # upper weight bound
    ):
        """
        :param conv:    The Conv2d layer whose weights will be updated.
        :param tau:     Exponential decay rate for synaptic traces (0 < τ < 1).
        :param A_plus:  LTP rate — how fast correlated connections strengthen.
        :param A_minus: LTD rate — how fast anti-correlated connections weaken.
        :param w_min:   Minimum allowed weight value.
        :param w_max:   Maximum allowed weight value.
        """
        self.conv    = conv
        self.tau     = tau
        self.A_plus  = A_plus
        self.A_minus = A_minus
        self.w_min   = w_min
        self.w_max   = w_max

        # Synaptic trace tensors — initialised on first call to update()
        self.pre_trace  = None   # shape: [B, C_in, H, W]
        self.post_trace = None   # shape: [B, C_out, H, W]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """
        Apply one STDP weight update step.

        :param pre_spikes:  Input spike tensor  [B, C_in,  H, W]  (DVS events)
        :param post_spikes: Output spike tensor [B, C_out, H, W]  (Conv-LIF output)
        """
        pre  = pre_spikes.detach().float()
        post = post_spikes.detach().float()

        # --- Initialise traces on first call ---
        if self.pre_trace is None:
            self.pre_trace  = torch.zeros_like(pre)
            self.post_trace = torch.zeros_like(post)

        # --- Update exponential traces ---
        # Each trace decays by τ each step and jumps up when a spike arrives.
        self.pre_trace  = self.tau * self.pre_trace  + pre
        self.post_trace = self.tau * self.post_trace + post

        # --- Compute weight updates (no autograd — purely Hebbian) ---
        with torch.no_grad():
            # LTP: pre was recently active AND post fires now → strengthen
            ltp = self._conv_weight_correlation(self.pre_trace, post)

            # LTD: pre fires now AND post was recently active → weaken
            ltd = self._conv_weight_correlation(pre, self.post_trace)

            dw = self.A_plus * ltp - self.A_minus * ltd
            self.conv.weight += dw

            # Clamp weights to biologically-motivated bounds [w_min, w_max]
            self.conv.weight.clamp_(self.w_min, self.w_max)

    def weight_norm(self) -> float:
        """Returns the L2 norm of the current conv weights — useful for logging."""
        return self.conv.weight.data.norm().item()

    def reset(self):
        """Resets traces (call when switching video sources)."""
        self.pre_trace  = None
        self.post_trace = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conv_weight_correlation(
        self,
        input_trace: torch.Tensor,
        output_trace: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the correlation between input and output traces projected
        onto the weight tensor shape [C_out, C_in, kH, kW].

        This is mathematically equivalent to the gradient of a Conv2d w.r.t.
        its weights, but driven by spike traces rather than a loss signal.

        Uses F.unfold to extract overlapping input patches and torch.einsum
        to efficiently compute the outer product across spatial positions.
        """
        kH, kW = self.conv.kernel_size
        pad    = self.conv.padding[0]

        B, C_in,  H,     W     = input_trace.shape
        _, C_out, H_out, W_out = output_trace.shape

        # Extract input patches: [B, C_in * kH * kW, H_out * W_out]
        patches = F.unfold(input_trace, kernel_size=kH, padding=pad)

        # Flatten output trace over spatial dims: [B, C_out, H_out * W_out]
        out_flat = output_trace.reshape(B, C_out, -1)

        # Compute outer product summed over batch and spatial dims:
        # result shape: [C_out, C_in * kH * kW]
        dw_flat = torch.einsum("bon,bin->oi", out_flat, patches)

        # Normalise by number of samples averaged
        dw_flat /= B * H_out * W_out

        return dw_flat.reshape(C_out, C_in, kH, kW)
