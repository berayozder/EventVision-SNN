import cv2
import numpy as np
import torch

class EventGenerator:
    """
    Simulates a Dynamic Vision Sensor (DVS) by detecting temporal
    log-luminance changes and converting them into ON/OFF spikes.

    Real DVS cameras respond to changes in log-luminance:
        ΔL = log(I_t) - log(I_{t-1})
    This gives high dynamic range and makes sensitivity scale-invariant
    (a candle in a dark room triggers the same ΔL as a floodlight in daylight).
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
        Processes an incoming BGR frame to produce binary spike maps
        using log-luminance differencing — matching real DVS physics.
        """
        # Convert to grayscale, normalize to [0, 1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Apply log transformation — epsilon avoids log(0) on black pixels
        # This is the core of DVS physics: the sensor is logarithmically sensitive
        log_gray = np.log(gray + 1e-6)

        if self.last_frame is None:
            self.last_frame = log_gray   # store log-luminance as reference
            return np.zeros_like(gray), np.zeros_like(gray)

        # ΔL = log(I_t) - log(I_{t-1})  →  threshold to get ON/OFF events
        diff = log_gray - self.last_frame

        # Generate spikes based on thresholding
        on_spikes  = (diff >  self.threshold).astype(np.float32)
        off_spikes = (diff < -self.threshold).astype(np.float32)

        # Store log-luminance for the next differential step
        self.last_frame = log_gray
        
        return on_spikes, off_spikes

    def convert_to_tensor(self, on_spikes, off_spikes):
        """
        Wraps the event data into a PyTorch tensor for snnTorch processing.
        Format: [Batch, Polarity (2), Height, Width]
        """
        # Stack channels: 0 for ON, 1 for OFF
        event_stack = np.stack([on_spikes, off_spikes], axis=0)
        return torch.from_numpy(event_stack).unsqueeze(0)