# EventVision-SNN

A project that makes a **normal webcam behave like a Dynamic Vision Sensor (DVS)** — a special camera used in robotics and neuromorphic computing.

A regular camera records a full picture every frame , even if nothing moves. A DVS is smarter: it only records **which pixels changed** and **whether they got brighter or darker**. This makes it much faster, uses less data, and only "wakes up" when something actually happens — similar to how your eye works.

This project **simulates** that behavior in software and then passes the detected changes through a small **Spiking Neural Network (SNN)** that learns to recognize edges and shapes on its own, without any labels.

---

## Two Modes

### Mode 1 — Real-time DVS demo (`src/main.py`)

Plug in your webcam (or point it at a video file) and watch the simulation run live. Three windows open side-by-side: the original camera feed, the detected "events" (pixel changes shown in color), and 8 feature maps showing what the SNN is currently detecting. No dataset needed.

### Mode 2 — N-MNIST digit classification (`scripts/run_nmnist.py`)

Uses the **N-MNIST** dataset — the classic handwritten-digits dataset (MNIST) re-recorded with a real DVS camera. The SNN trains on this data using STDP (no labels at all), then its accuracy is measured with a simple Winner-Take-All readout.

---

## What Happens Step-by-Step (Mode 1)

1. **Read a frame** from your webcam or video file.
2. **Detect pixel changes** by comparing the log-brightness of the current frame with the previous one. Pixels that changed more than a threshold become ON events (got brighter) or OFF events (got darker).
3. **Pass the events** through a Conv2D layer with 8 small 3×3 kernels that each look for a different edge orientation (horizontal, vertical, diagonal, etc.).
4. **Fire LIF neurons** — each neuron has a "charge bucket" that fills up when edges are detected and slowly leaks away. When it overflows, the neuron fires.
5. **Update the kernels** using STDP after every frame. Kernels that responded to the same events at the same time get stronger. No teacher, no labels.
6. **Show three windows** and print a summary every 30 frames.

```
[Frame    30] Weight norm: 1.2140 | Sparsity: 97.4%
```

---

## Key Concepts

| Term | What it means |
|---|---|
| **DVS / Event camera** | A camera that only reports *changes* in brightness, not full frames |
| **ON / OFF event** | A pixel "raising its hand" — green = got brighter, red = got darker |
| **Log-luminance** | Brightness measured on a log scale. `ΔL = log(I_t) − log(I_{t−1})`. Matches how real DVS cameras and human eyes work — doubling the brightness always triggers the same size event |
| **LIF Neuron** | Leaky Integrate-and-Fire. Think of a bucket with a tiny hole: input fills it, the leak drains it, and when it overflows the neuron fires and resets |
| **STDP** | Spike-Timing Dependent Plasticity. The brain's learning rule: if one neuron fires just before another, their connection strengthens. No backprop needed |
| **Gabor filter** | A small striped patch pattern. It's the mathematical model of how the primary visual cortex (V1) detects oriented edges. The 8 kernels are initialized as Gabor filters so the network is useful from the very first frame |
| **Spike sparsity** | At any moment only ~1–5% of neurons fire. This is the key energy argument for neuromorphic hardware: if 99% of neurons are silent, 99% of the work is skipped |
| **Winner-Take-All (WTA)** | A simple readout rule: whichever output neuron fires most "wins" and its assigned class is the prediction |

---

## SNN Architecture

The network is a one-layer **V1-cortex-inspired** pipeline:

```
Input Events [1, 2, H, W]         ← batch=1, 2 channels: ON and OFF
      │
      ▼
Conv2D  (8 kernels, 3×3)           ← scans the event map for edges
      │   initialized as Gabor filters (8 orientations, 0° → 157.5°)
      ▼
LIF Neurons  (one per pixel)       ← fires when a detected edge is strong enough
      │   beta=0.8 (membrane decay), threshold=1.0
      ▼
Output Spikes [1, 8, H, W]         ← 8 maps showing WHERE each edge type is active
      │
      ▼
STDP Update                        ← adjusts Conv2D weights using synaptic traces
                                     no labels, no gradients, fully online
```

---

## Project Structure

```
EventVision-SNN/
├── src/
│   ├── main.py         Mode 1 entry point — webcam / video DVS emulator + live display
│   ├── generator.py    EventGenerator — converts video frames into ON/OFF spike maps
│   ├── processor.py    SNNProcessor — Conv2D + Gabor init + LIF neurons
│   ├── stdp.py         STDPLearner — updates Conv2D weights using synaptic traces
│   ├── dataset.py      N-MNIST loader — event streams → time-binned spike frames
│   ├── utils.py        visualize_events() and visualize_feature_maps() helpers
│   └── __init__.py
├── scripts/
│   ├── run_nmnist.py   Mode 2 entry point — STDP training + WTA classification
│   └── verify_mnist.py Utility — checks that the Kaggle MNIST IDX files are valid
├── tests/
│   ├── test_pipeline.py   Integration tests for the full DVS → SNN pipeline
│   ├── test_stdp.py       Unit tests for the STDP weight update rule
│   └── test_dataset.py    Unit tests for the N-MNIST data loader
├── data/
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`): `opencv-python`, `numpy`, `torch`, `snntorch`, `matplotlib`, `tonic`

---

## Run Mode 1 — Real-time DVS Demo

```bash
# From your webcam:
python src/main.py

# From a video file:
python src/main.py --source data/test_vid1.mp4
```

Three windows open:
- **Original Stream** — camera feed with a live **Sparsity %** overlay
- **Event-Based (DVS) Spikes** — green = ON events, red = OFF events
- **SNN Feature Maps (8 Edge Detectors)** — 2×4 grid of the 8 conv kernel responses

The console prints a status line every 30 frames:
```
[Frame    30] Weight norm: 1.2140 | Sparsity: 97.4%
```
- **Weight norm rising** → STDP is actively learning
- **Sparsity ~95–99%** → biologically realistic and energy-efficient

Press **`q`** to quit. If a video file ends, the demo loops automatically.

---

## Run Mode 2 — N-MNIST Classification

N-MNIST (~180 MB) is downloaded automatically on the first run via the `tonic` library.

```bash
# Quick test (1000 training samples):
python scripts/run_nmnist.py --n_train 1000

# Full run (5000 train / 1000 test):
python scripts/run_nmnist.py

# Train only (saves weights to data/trained_weights.pt):
python scripts/run_nmnist.py --mode train

# Evaluate only (loads saved weights):
python scripts/run_nmnist.py --mode eval
```

**What happens:**
1. **Phase 1 — STDP Training**: Each digit sample is fed frame-by-frame through the SNN. STDP updates the Conv2D kernels online. No labels used.
2. **Phase 2 — WTA Evaluation**: Each output neuron is assigned to its most-responsive class. Test samples are classified by whichever neuron fires most. Overall accuracy and per-class breakdown are printed.

Sample output:
```
  Class    Correct    Total      Acc
  ------------------------------------
  0            87      100    87.0%
  1            91      100    91.0%
  ...
  ====================================
  Overall WTA Accuracy: 65.3%  (chance = 10%)
```

> Chance level is 10% (10 classes). A score well above that confirms the STDP kernels learned meaningful features without any supervision.

After training finishes, a **kernel evolution plot** is automatically saved to `data/kernel_evolution.png` and displayed on screen. It shows all 8 Conv2D kernels in two rows — **before** (Gabor filters) and **after** (STDP-tuned) — using a diverging colormap so you can see exactly how each kernel morphed to fit the curves of handwritten digits.

---

## Run Tests

```bash
python -m pytest tests/ -v
```

Three test files cover:
- `test_pipeline.py` — end-to-end: frame → events → SNN spikes → STDP update
- `test_stdp.py` — STDP weight update math (LTP, LTD, clamping, traces)
- `test_dataset.py` — N-MNIST loader output shapes and dtypes