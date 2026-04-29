# Epilepsy Seizure Detection — SNN on FPGA

A real-time, wearable-friendly epilepsy seizure detection system implemented as a 2-layer **Spiking Neural Network (SNN)** in synthesizable Verilog, targeting the **Xilinx Zynq-7020 FPGA**. The network processes 4-channel EEG data using an amplitude-based one-hot spike encoder and classifies seizure vs. normal activity through two dense LIF (Leaky Integrate-and-Fire) layers.

---

## Implementation Summary

<table>
  <thead>
    <tr style="background-color:#0d1117; color:#58a6ff;">
      <th colspan="2" align="center"
          style="font-size:1.15em; padding:12px 16px; letter-spacing:0.04em;">
        ⚙️ Implementation Specifications
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Target Platform</strong></td>
      <td><strong>Xilinx Zynq-7020 FPGA</strong> (xc7z020clg484-1)</td>
    </tr>
    <tr>
      <td><strong>RTL Language</strong></td>
      <td>Verilog (IEEE 1364-2005)</td>
    </tr>
    <tr>
      <td><strong>Design &amp; Simulation Tool</strong></td>
      <td>Xilinx Vivado 2025.2 / XSim</td>
    </tr>
    <tr>
      <td><strong>System Clock</strong></td>
      <td>50 MHz (20 ns period)</td>
    </tr>
    <tr>
      <td><strong>Inference Latency</strong></td>
      <td>5 clock cycles — 100 ns @ 50 MHz</td>
    </tr>
    <tr>
      <td><strong>Pipeline Stages</strong></td>
      <td>IDLE → ENCODE → LAYER1 → LAYER2 → DECODE</td>
    </tr>
    <tr>
      <td><strong>SNN Architecture</strong></td>
      <td>64 → 8 → 1 (2 dense LIF layers)</td>
    </tr>
    <tr>
      <td><strong>EEG Input Channels</strong></td>
      <td>4 temporal channels (16-bit signed per channel)</td>
    </tr>
    <tr>
      <td><strong>Encoding Scheme</strong></td>
      <td>One-hot amplitude — 16 levels/channel → 64-bit spike bus<br>
          (exactly 4 active spikes per sample)</td>
    </tr>
    <tr>
      <td><strong>Neuron Model</strong></td>
      <td>LIF with multiplicative decay — α = 0.9375 (DECAY_SHIFT = 4)</td>
    </tr>
    <tr>
      <td><strong>Weight Precision</strong></td>
      <td>INT8 (8-bit signed)</td>
    </tr>
    <tr>
      <td><strong>Membrane Precision</strong></td>
      <td>INT16 (16-bit signed, saturation-clamped)</td>
    </tr>
    <tr>
      <td><strong>Total Trainable Weights</strong></td>
      <td>520 (Layer-1: 512 · Layer-2: 8)</td>
    </tr>
    <tr>
      <td><strong>Non-zero Weights</strong></td>
      <td>491 (484 + 7) — 94.4% weight density</td>
    </tr>
    <tr>
      <td><strong>Decoder</strong></td>
      <td>Sliding-window firing-rate counter<br>
          Window = 4 096 samples · Blanking = 256 · Threshold = 1 500 spikes</td>
    </tr>
    <tr>
      <td><strong>Training Dataset</strong></td>
      <td>Bonn EEG Dataset — classes Z, O, N (normal) vs. S (seizure)</td>
    </tr>
    <tr>
      <td><strong>Total Training Samples</strong></td>
      <td>409 700 (307 275 normal + 102 425 seizure)</td>
    </tr>
    <tr>
      <td><strong>Validation Accuracy</strong></td>
      <td><strong>94.78 %</strong> (train 94.57 %)</td>
    </tr>
    <tr>
      <td><strong>Simulation Sign-off</strong></td>
      <td>✅ Functional simulation passed (XSim)</td>
    </tr>
  </tbody>
</table>

---

## Repository Structure

```
Epilepsy_Seizure_Detection/
├── EEG-Dataset/                    # Raw Bonn EEG data (classes Z, O, N, S)
├── Epilepsy_Seizure_Detection_verilog/
│   ├── snn_seizure_top.v           # Top-level pipeline FSM
│   ├── eeg_amplitude_encoder.v     # 4-ch EEG → 64-bit one-hot spike bus
│   ├── dense_layer.v               # Dense LIF layer (N neurons)
│   ├── lif_neuron.v                # Leaky Integrate-and-Fire neuron
│   ├── firing_rate_decoder.v       # Sliding-window seizure decoder
│   ├── snn_seizure_tb.v            # Testbench
│   ├── constraints.xdc             # Timing constraints (50 MHz, Zynq-7020)
│   ├── layer1_weights.mem          # Quantised INT8 Layer-1 weights
│   ├── layer2_weights.mem          # Quantised INT8 Layer-2 weights
│   ├── eeg_dataset.mem             # Encoded EEG stimulus for simulation
│   └── training_metrics.json       # Accuracy, TP/TN/FP/FN, weight stats
├── SNN_traning.py                  # ANN surrogate training + INT8 weight export
├── EEG_mem.py                      # Dataset loader & .mem file generator
└── README.md
```

---

## Quick Start

### 1 — Train and export weights
```bash
python SNN_traning.py --dataset_root EEG-Dataset/Dataset \
                      --output_dir Epilepsy_Seizure_Detection_verilog
```

### 2 — Simulate in Vivado XSim
Open `Epilepsy_Seizure_Detection_verilog.xpr` in **Vivado 2025.2**, set
`snn_seizure_tb` as the top simulation source, and run the XSim flow.

---

## Network Architecture

```
 EEG Input (4 ch × 16-bit)
        │
 ┌──────▼──────────────────────┐
 │  EEG Amplitude Encoder      │  4 ch → 64-bit one-hot spike bus
 │  (16 amplitude levels/ch)   │  (exactly 4 active bits/sample)
 └──────┬──────────────────────┘
        │  64-bit spike bus
 ┌──────▼──────────────────────┐
 │  Dense Layer 1              │  64 inputs → 8 LIF neurons
 │  Threshold = 200            │  INT8 weights · INT16 membrane
 │  Decay α = 0.9375           │  Soft-reset on spike
 └──────┬──────────────────────┘
        │  8-bit spike bus
 ┌──────▼──────────────────────┐
 │  Dense Layer 2              │  8 inputs → 1 LIF neuron
 │  Threshold = 100            │  INT8 weights · INT16 membrane
 │  Decay α = 0.9375           │
 └──────┬──────────────────────┘
        │  1-bit spike
 ┌──────▼──────────────────────┐
 │  Firing-Rate Decoder        │  Sliding window (4 096 samples)
 │  Seizure threshold = 1 500  │  Post-ictal blanking = 256
 └──────┬──────────────────────┘
        │
  seizure_alert (1-bit)
```

---

## Reference

> P. Busia, G. Leone, A. Matticola, L. Raffo, P. Meloni,
> *"Wearable Epilepsy Seizure Detection on FPGA With Spiking Neural Networks,"*
> **IEEE Transactions on Biomedical Circuits and Systems**, vol. 19, no. 6, pp. 1175–, Dec. 2025.
> DOI: [10.1109/TBCAS.2025.3575327](https://doi.org/10.1109/TBCAS.2025.3575327)
