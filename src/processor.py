import torch
import snntorch as snn

class SNNProcessor:
    """
    A Spiking Neural Network layer that processes DVS events 
    using Leaky Integrate-and-Fire (LIF) neurons.
    """
    def __init__(self, beta=0.9, threshold=1.0):
        """
        :param beta: Decay rate of the membrane potential (0.0 < beta < 1.0).
        :param threshold: The potential at which a neuron fires its own spike.
        """
        # Initialize the LIF neuron
        # beta defines the 'leak'—how fast the neuron forgets past input
        self.lif = snn.Leaky(beta=beta, threshold=threshold)
        self.mem = None  # Membrane potential state

    def process(self, event_tensor):
        """
        Passes the event tensor through the LIF neuron.
        :param event_tensor: Tensor of shape [Batch, Polarity, H, W]
        :return: spk (Output spikes), self.mem (Updated membrane potential)
        """
        # Initialize membrane potential on the first pass
        if self.mem is None:
            self.mem = self.lif.init_leaky()
            # Ensure memory matches the input shape [Batch, Polarity, H, W]
            # We expand the default 1D state to match the spatial dimensions
            self.mem = torch.zeros_like(event_tensor)

        # Forward pass: current events + previous memory = new spikes & new memory
        # spk is 1 where the membrane potential exceeded the threshold
        spk, self.mem = self.lif(event_tensor, self.mem)
        
        return spk, self.mem

    def reset(self):
        """Resets the internal state of the neurons."""
        self.mem = None