# EventVision-SNN

Simulates a **Dynamic Vision Sensor (DVS)** using a regular webcam, then processes the output through a **Spiking Neural Network (SNN)** that learns edge detectors on its own — no labels needed.

---

## What is a DVS?

A regular camera captures full frames at fixed intervals. A **DVS (event camera)** only records *which pixels changed* and *in which direction* — ignoring everything that stayed the same. This makes it extremely fast and power-efficient.

This project emulates that behaviour in software: it compares consecutive webcam frames and converts brightness changes into **ON events** (pixel got brighter) and **OFF events** (pixel got darker).

---

## How It Works

```
Webcam frame
      │
      ▼
DVS emulator       ─── ON/OFF spike maps  [H, W]
      │
      ▼
Conv2D (8 × 3×3)   ─── initialized as Gabor filters (edge detectors)
      │
      ▼
LIF neurons        ─── fire when membrane potential crosses threshold
      │
      ▼
STDP update        ─── strengthens connections that activated together
```

---

## Key Terms

| Term | Meaning |
|---|---|
| **DVS / Event camera** | Camera that only reports pixel-level brightness *changes* |
| **ON / OFF event** | A pixel "raising its hand" — brighter or darker than the last frame |
| **LIF neuron** | Leaky Integrate-and-Fire: charge accumulates, leaks away, fires when full |
| **STDP** | Spike-Timing Dependent Plasticity — neurons that fire together, wire together |
| **Gabor filter** | Mathematical model of edge-detecting cells in the primary visual cortex (V1) |
| **Spike sparsity** | At any moment, ~1–5% of neurons fire — the rest are silent and use no energy |

---

## Project Structure

```
EventVision-SNN/
├── src/
│   ├── main.py         Entry point — live webcam DVS demo
│   ├── generator.py    EventGenerator — frame → ON/OFF spike maps
│   ├── processor.py    SNNProcessor — Conv2D + LIF neurons
│   ├── stdp.py         STDPLearner — online weight updates
│   └── utils.py        Visualisation helpers
├── tests/
│   ├── test_pipeline.py
│   └── test_stdp.py
├── data/
├── requirements.txt
└── .gitignore
```

> **N-MNIST classification experiment** lives on the [`experiments/nmnist`](../../tree/experiments/nmnist) branch.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Run

```bash
# Webcam
python src/main.py

# Video file
python src/main.py --source data/test_vid1.mp4
```

Three windows open:
- **Original feed** — raw camera output
- **DVS events** — green = ON, red = OFF
- **SNN feature maps** — 8 edge detectors responding in real time

Press **`q`** to quit. Status prints every 30 frames:
```
[Frame    30] Weight norm: 1.2140 | Sparsity: 97.4%
```

---

## Run Tests

```bash
python -m pytest tests/ -v
```