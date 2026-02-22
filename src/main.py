import cv2
import argparse
from generator import EventGenerator
from processor import SNNProcessor
from utils import visualize_events


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
            mem_vis = mem[0, 0].detach().numpy()  # Take ON-channel potential

            cv2.imshow('Original Stream', frame)
            cv2.imshow('Event-Based (DVS) Spikes', event_vis)
            cv2.imshow('SNN Membrane Potential (Memory)', mem_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()