import cv2
import numpy as np
import torch

class EventGenerator:
    """
    Simulates a Dynamic Vision Sensor (DVS) by detecting temporal 
    brightness changes and converting them into ON/OFF spikes.
    """
    def __init__(self, threshold=0.15):
        """
        :param threshold: Sensitivity of the event trigger. 
                          Lower values capture more subtle movements.
        """
        self.threshold = threshold
        self.last_frame = None

    def process_frame(self, frame):
        """
        Processes an incoming BGR frame to produce binary spike maps.
        """
        # Convert to grayscale and normalize to [0, 1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        if self.last_frame is None:
            self.last_frame = gray
            return np.zeros_like(gray), np.zeros_like(gray)

        # Calculate temporal difference (approx. delta log intensity)
        diff = gray - self.last_frame
        
        # Generate spikes based on thresholding
        on_spikes = (diff > self.threshold).astype(np.float32)
        off_spikes = (diff < -self.threshold).astype(np.float32)

        # Update state for the next differential step
        self.last_frame = gray
        
        return on_spikes, off_spikes

    def convert_to_tensor(self, on_spikes, off_spikes):
        """
        Wraps the event data into a PyTorch tensor for snnTorch processing.
        Format: [Batch, Polarity (2), Height, Width]
        """
        # Stack channels: 0 for ON, 1 for OFF
        event_stack = np.stack([on_spikes, off_spikes], axis=0)
        return torch.from_numpy(event_stack).unsqueeze(0)