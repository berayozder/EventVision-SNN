# EventVision-SNN

A project that makes a normal webcam behave like a **Dynamic Vision Sensor (DVS)** — a type of smart camera used in robotics and neuromorphic computing.

Instead of capturing full frames at a fixed frame rate (like a regular camera), a DVS only records **which pixels changed** and **in which direction** (brighter or darker). This makes it much faster and more data-efficient.

![Demo — DVS emulator running on a video file](demo.gif)

---

## What Does This Project Do?

1. **Reads frames** from your webcam (or a video file)
2. **Detects pixel changes** between frames — if a pixel got brighter, it fires an **ON event**; if it got darker, it fires an **OFF event**
3. **Passes those events** through a simple Spiking Neural Network (SNN), which mimics how biological neurons process information over time
4. **Shows you** three live windows: the original feed, the detected events, and the neuron activity

---

## Core Ideas

**Threshold** — A sensitivity knob. Only changes bigger than this value trigger an event. Lower = more sensitive but noisier.

**ON / OFF Events** — Think of them as pixels raising their hand when something moves. Green = got brighter, Red = got darker.

**Leaky Integrate-and-Fire (LIF) Neuron** — Each pixel has a "charge bucket" that fills up when events arrive and slowly drains over time. When it overflows, the neuron "fires". It's the simplest model of a biological neuron.

---

## Project Structure

```
EventVision-SNN/
├── src/
│   ├── generator.py   # Converts webcam frames into ON/OFF spike maps
│   ├── processor.py   # Runs those spikes through LIF neurons (snnTorch)
│   ├── utils.py       # Helper: makes a colored image of the spikes
│   └── main.py        # Entry point — ties everything together
├── tests/
│   └── test_pipeline.py  # Automated tests (no camera needed)
├── data/              # Put video files here (ignored by git)
└── requirements.txt
```

---

## Setup

```bash
# Clone the repo and create a virtual environment
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Run

```bash
# Activate your virtual environment first, then:
python src/main.py
# or with a video file:
python src/main.py --source data/test_vid1.mp4
```

Three windows will open:
- **Original Stream** — your webcam feed or video file
- **Event-Based (DVS) Spikes** — green pixels = brightness increased, red = decreased
- **SNN Membrane Potential** — how much "charge" each neuron has built up

Press **`q`** to quit.

---

## Run Tests

```bash
python -m pytest tests/ -v
```