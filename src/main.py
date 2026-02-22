import cv2
import argparse
from generator import EventGenerator
from processor import SNNProcessor
from utils import visualize_events, visualize_feature_maps
from stdp import STDPLearner


def main():
    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="DVS Emulator — webcam or video file")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to a video file (e.g. data/clip.mp4). Omit to use the webcam."
    )
    args = parser.parse_args()

    # Use the file path if given, otherwise fall back to webcam (device 0)
    source = args.source if args.source else 0
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: could not open source '{source}'.")
        return

    # Set the sensitivity (lower threshold = noisier / more sensitive)
    generator = EventGenerator(threshold=0.1)

    # Higher beta = longer memory of motion
    processor = SNNProcessor(beta=0.8, threshold=1.0)

    # STDP learner — updates Conv2D kernels online based on spike correlations
    # A_plus / A_minus control potentiation vs depression rates
    stdp = STDPLearner(processor.conv, tau=0.9, A_plus=0.005, A_minus=0.005)

    source_label = args.source if args.source else "webcam"
    print(f"DVS Simulation starting on '{source_label}'... Press 'q' to quit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # End of video file — loop back to the start
                if args.source:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    generator = EventGenerator(threshold=0.1)  # reset reference frame
                    processor.reset()
                    stdp.reset()
                    continue
                break

            # 1. Event Generation
            on_spikes, off_spikes = generator.process_frame(frame)

            # 2. Convert to Tensor for snnTorch
            event_tensor = generator.convert_to_tensor(on_spikes, off_spikes)

            # 3. Process through SNN
            # spk: binary spikes fired by neurons
            # mem: membrane potential — the neuron's running "memory"
            spk, mem = processor.process(event_tensor)

            # 4. Visualization
            event_vis = visualize_events(on_spikes, off_spikes)
            feat_vis = visualize_feature_maps(mem)  # tiled 2×4 grid of 8 edge detectors

            cv2.imshow('Original Stream', frame)
            cv2.imshow('Event-Based (DVS) Spikes', event_vis)
            cv2.imshow('SNN Feature Maps (8 Edge Detectors)', feat_vis)

            # 5. STDP weight update — kernels learn from spike correlations
            stdp.update(event_tensor, spk)

            # Log weight norm every 30 frames so we can see learning happening
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_idx % 30 == 0:
                print(f"[Frame {frame_idx:5d}] Weight norm: {stdp.weight_norm():.4f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()