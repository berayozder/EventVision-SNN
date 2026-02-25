# Neuromorphic Computing — Concepts Behind EventVision-SNN

A field guide for a computer engineering student who wants to understand how this project works — and why it matters for the future of computing.

---

## Table of Contents

1. [The Big Picture — What is Neuromorphic Computing?](#1-the-big-picture)
2. [DVS — the Event Camera](#2-dvs--the-event-camera)
3. [ON/OFF Events and Log-Luminance](#3-onoff-events-and-log-luminance)
4. [Gabor Filters — How Brains Detect Edges](#4-gabor-filters)
5. [Convolutional Layer — Scanning for Patterns](#5-convolutional-layer)
6. [LIF Neurons — The Leaky Bucket](#6-lif-neurons--the-leaky-bucket)
7. [STDP — Learning Without Labels](#7-stdp--learning-without-labels)
8. [Spike Sparsity — Why This Saves Energy](#8-spike-sparsity)
9. [How It All Fits Together](#9-how-it-all-fits-together)
10. [What This Project Demonstrates](#10-what-this-project-demonstrates)

---

## 1. The Big Picture

### Why not just use a regular neural network?

Deep learning (CNNs, transformers) works very well — but it has a problem: it consumes enormous amounts of power. Training GPT-4 used roughly as much electricity as 500 transatlantic flights. Even *running* a modern model takes hundreds of watts.

The brain, by contrast, runs the most complex cognition known to science on about **20 watts** — the same as a dim light bulb.

**Neuromorphic computing** is the study of how to build hardware and algorithms that work more like the brain:

- Neurons are **silent by default** (use no energy)
- They only "fire" when something meaningful happens
- Learning is **local** (each synapse updates itself based only on what it sees)
- There is no global clock — events happen when they happen

This project is a software simulation of these principles using a webcam.

---

## 2. DVS — the Event Camera

### Regular camera vs. DVS

| | Regular Camera | DVS (Dynamic Vision Sensor) |
|---|---|---|
| **Output** | Full pixel grid every N ms | Individual pixel events as they happen |
| **Latency** | ~33 ms (30 fps) | ~1 µs per event |
| **Data rate** | Every pixel, every frame | Only changed pixels |
| **Power** | High (always reading everything) | Very low (silent when nothing moves) |

### How a DVS pixel works

Each pixel in a DVS has its own tiny circuit that watches its own brightness. When the brightness changes by more than a threshold, that single pixel sends an event:

```
Pixel brightness rises  →  ON event  (something got brighter here)
Pixel brightness falls  →  OFF event (something got darker here)
Nothing changed        →  silence   (no data sent at all)
```

A DVS camera attached to a moving robot generates ~1 million events/second in a busy scene, but drops to near zero in a static room. A regular camera would keep sending all 1.2 million pixels at 30 fps regardless.

### How this project emulates a DVS

We don't have a real DVS chip. Instead, `generator.py` does this every frame:

1. Convert the RGB frame to **log-luminance** (see Section 3)
2. Subtract the previous frame's log-luminance from the current one → `ΔL`
3. Pixels where `ΔL > +threshold` → **ON event**
4. Pixels where `ΔL < -threshold` → **OFF event**
5. Everything else → silence

The result is a software "event stream" that behaves like a real DVS.

---

## 3. ON/OFF Events and Log-Luminance

### Why log-luminance?

Brightness (luminance) varies enormously — from a dark room (~1 lux) to direct sunlight (~100,000 lux). If you simply subtract raw pixel values, a small change in a dark scene and the same physical change in a bright scene would give very different numbers, making your threshold useless.

Real DVS cameras — and the human eye — respond to **relative** changes in brightness. The maths behind this is:

```
ΔL = log(I_current) - log(I_previous) = log(I_current / I_previous)
```

This means: *"the brightness doubled"* always gives the same `ΔL` regardless of whether you're in a dim or bright room. A 2× change in a dark room and a 2× change in a bright room both look the same. This is called **logarithmic sensitivity** and it's what makes your eyes work well from candlelight to noon sun.

In code (`generator.py`):

```python
log_frame = np.log1p(gray.astype(np.float32))  # log(1 + pixel_value)
delta = log_frame - self.prev_log_frame
on_events  = (delta >  self.threshold).astype(np.float32)
off_events = (delta < -self.threshold).astype(np.float32)
```

### What the two channels mean

The SNN receives events as a tensor of shape `[1, 2, H, W]`:
- **Channel 0** = ON map (1 where a pixel brightened, 0 elsewhere)
- **Channel 1** = OFF map (1 where a pixel darkened, 0 elsewhere)

This two-channel representation is exactly what a real DVS sends. The convolution layer then processes both channels simultaneously.

---

## 4. Gabor Filters

### What is a Gabor filter?

A Gabor filter is a mathematical pattern that looks like a striped wave. It was invented to model how neurons in the brain's primary visual cortex (called **V1**) detect edges.

Imagine a filter that is bright-dark-bright in stripes running at 45°. When you slide it over an image:
- Where there's an edge at 45°, the filter response is **high**
- Where the image is flat or the edge runs at a different angle, the response is **low**

By creating 8 Gabor filters at angles 0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5° — you cover all possible edge orientations.

### Why initialize the convolutional kernels as Gabors?

The Conv2D layer has 8 kernels (3×3 each). These kernels are what the network "looks through" to detect patterns. Instead of starting from random noise, we initialize them as Gabor filters because:

1. **They work immediately** — even before any learning, the network produces useful edge maps
2. **They match biology** — mammalian V1 neurons are known to be well-described by Gabor functions
3. **STDP doesn't need to discover from scratch** — it only needs to *refine* the filters, not invent them

In code (`processor.py`), `cv2.getGaborKernel()` generates these for each of the 8 orientations and they are written into `self.conv.weight.data`.

---

## 5. Convolutional Layer

### Quick review

A convolutional layer slides a small kernel (filter) over the input image, computing a dot product at each position. The output is called a **feature map** — it's a grid whose value at each position says "how strongly does this kernel match what's at this location?"

```
Input event map: [1, 2, H, W]   (batch=1, ON+OFF channels)
        │
        ▼  8 kernels, each 3×3, applied to both channels
        ▼
Feature maps:    [1, 8, H, W]   (one map per kernel)
```

Each of the 8 output maps answers a different question: *"Where are the 0° edges?"*, *"Where are the 45° edges?"* etc.

### This is the only weight matrix in the network

Everything that is "learned" in this project lives in these 8×2×3×3 = 144 weights. STDP updates only these weights. No other parameters change.

---

## 6. LIF Neurons — the Leaky Bucket

### The analogy

Imagine a bucket with a small hole at the bottom:
- Water poured in = **synaptic input** (the convolutional feature map output)
- Water slowly leaking out = **membrane leak** (decay)
- Bucket overflowing = **neuron firing** (a spike)
- Bucket instantly empties after overflow = **reset**

This is the **Leaky Integrate-and-Fire (LIF)** neuron model.

### The maths

At each time step `t`, the membrane potential `V(t)` evolves as:

```
V(t) = β · V(t-1) + I(t)
```

Where:
- `β` (beta) is the **leak factor**, between 0 and 1. `β = 0.8` means 20% of the charge leaks away each step.
- `I(t)` is the **input current** — the convolution output at this time step.

When `V(t) ≥ threshold` (default 1.0):
- The neuron **fires** (outputs a spike = 1)
- `V` resets to 0

When `V(t) < threshold`:
- The neuron is **silent** (outputs 0)
- `V` carries over to the next step, modified by leak

### Why not just use ReLU?

ReLU (`max(0, x)`) fires a continuous value proportional to input. LIF fires a **binary spike** (0 or 1) and only when the accumulated input exceeds a threshold. This has two important properties:

1. **Temporal integration** — weak inputs over time can eventually fire the neuron. ReLU ignores history.
2. **Sparsity** — most neurons are silent most of the time, which is the key to energy efficiency.

In code, `snntorch.Leaky` implements this. `processor.py` creates one LIF layer that matches the spatial size of the feature maps.

---

## 7. STDP — Learning Without Labels

### The biological rule

In 1949, Donald Hebb proposed: *"Neurons that fire together, wire together."* STDP (Spike-Timing Dependent Plasticity) is the experimentally measured version of this rule.

The key insight: **causality determines the sign of the weight change**.

```
If pre-synaptic neuron fires JUST BEFORE post-synaptic neuron:
    → connection is STRENGTHENED  (Long-Term Potentiation, LTP)
    → the pre caused the post

If pre-synaptic neuron fires JUST AFTER post-synaptic neuron:
    → connection is WEAKENED  (Long-Term Depression, LTD)
    → the pre didn't contribute to the post's firing
```

### How it's implemented here

Because we don't track exact spike times, we use **synaptic traces** — a running exponential average of recent activity.

Each neuron maintains a trace that:
- **Jumps up** when that neuron spikes
- **Decays exponentially** between spikes at rate `tau = 0.9`

```
trace(t) = tau · trace(t-1) + spike(t)
```

The weight update at each time step (`stdp.py`):

```python
# LTP: pre fired, post is firing now → strengthen
ΔW_ltp = A_plus  * post_spk * pre_trace

# LTD: post fired, pre is firing now → weaken
ΔW_ltd = A_minus * pre_input * post_trace

ΔW = ΔW_ltp - ΔW_ltd
```

Where:
- `pre_trace` = how recently the input (event map) was active
- `post_trace` = how recently the output (spike map) was active
- `A_plus` / `A_minus` = learning rate constants (both 0.005 here)

### Why no labels?

STDP is **unsupervised**. It doesn't know what the input "means" — it only knows which inputs tended to co-activate with which outputs. Over many samples:
- Inputs that consistently co-occur (e.g. vertical edges) drive the same output neurons
- Those connections get stronger
- The kernel "learns" to be selective for that pattern

This is why the 8 kernels, initialized as Gabor edge detectors, should become more tuned to the specific edge statistics of the input stream over time.

---

## 8. Spike Sparsity

### What sparsity means

At any moment, only ~1–5% of neurons fire. The other 95–99% output exactly zero.

This looks like a disadvantage. It is actually the **entire energy argument** for neuromorphic hardware.

### Why it saves energy

In conventional hardware (CPUs/GPUs), every multiply-accumulate (MAC) operation costs energy whether or not the operand is zero.

In neuromorphic hardware (Intel Loihi, IBM TrueNorth, BrainScaleS):
- A neuron that isn't firing sends **no signal**
- No signal = no computation downstream
- No computation = **no energy consumed**

If 99% of neurons are silent → 99% of the MACs are skipped → ~100× energy reduction.

This is why the **sparsity** metric printed every 30 frames matters:
```
[Frame    30] Weight norm: 1.2140 | Sparsity: 97.4%
```
`97.4%` means only 2.6% of neurons fired that frame. On a real neuromorphic chip, that frame would use ~38× less energy than if all neurons fired.

### The threshold controls sparsity

The LIF threshold determines how hard it is to make a neuron fire:
- **Low threshold** → neurons fire easily → dense activity → less sparse
- **High threshold** → neurons fire rarely → very sparse, but might miss weak signals

`threshold = 1.0` with `beta = 0.8` is tuned to give biologically realistic sparsity (~95–99%).

---

## 9. How It All Fits Together

Here is the full data flow for one webcam frame, step by step:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Webcam frame arrives (BGR, uint8, 480×640)               │
└───────────────────────┬─────────────────────────────────────┘
                        │ generator.py
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Convert to log-luminance, subtract previous frame        │
│    Output: ON map [H,W], OFF map [H,W]  (binary 0/1)        │
└───────────────────────┬─────────────────────────────────────┘
                        │ stack into [1, 2, H, W] tensor
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Conv2D with 8 Gabor-initialized 3×3 kernels              │
│    Each kernel outputs a feature map showing WHERE           │
│    its preferred edge orientation appears                    │
│    Output: [1, 8, H, W]  (8 edge maps, continuous values)   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. LIF neurons (one per feature map pixel)                  │
│    Membrane = beta * prev_membrane + feature_map_value      │
│    If membrane >= 1.0 → spike (1), reset                    │
│    Output: spikes [1, 8, H, W]  (binary 0/1)                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. STDP update                                              │
│    Update pre_trace and post_trace                          │
│    ΔW = A+ * post_spk * pre_trace - A- * pre_input * post_trace│
│    Conv2D weights nudged, clamped to [−1, 1]                │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Visualisation                                            │
│    - events shown as green/red pixels                       │
│    - 8 feature maps shown in a 2×4 grid                     │
└─────────────────────────────────────────────────────────────┘
```

### The feedback loop over time

After thousands of frames, STDP has seen many edges. The kernels that started as generic Gabors drift to become tuned to **the specific edge statistics of this particular input stream**. If your webcam faces a scene full of horizontal lines (e.g. venetian blinds), the horizontal-edge kernels will strengthen and the others may weaken.

There is no explicit objective — no loss function. The network self-organises based purely on correlation in the input.

---

## 10. What This Project Demonstrates

| Concept | Where it appears | Why it matters |
|---|---|---|
| **Event-based sensing** | `generator.py` | Sparse data → less bandwidth, lower latency |
| **Log-luminance** | `generator.py` | Invariant to absolute brightness (like real DVS) |
| **Gabor initialization** | `processor.py` | Biological prior, useful from frame 1 |
| **Conv2D spatial processing** | `processor.py` | Detects local patterns (edges) across the whole image |
| **LIF temporal integration** | `processor.py` | Neurons accumulate evidence over time before firing |
| **Binary spikes** | `processor.py` | Enables sparsity; the key to neuromorphic energy savings |
| **STDP unsupervised learning** | `stdp.py` | Kernels self-organise without labels or backprop |
| **Synaptic traces** | `stdp.py` | Efficient approximation of spike-timing differences |
| **Spike sparsity** | Output ~97% | Proof of energy-efficient computation |

