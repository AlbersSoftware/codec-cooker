# Codec-Cooker

> **An interactive image and video compression research workbench for experimenting with transform coding, quantization, entropy coding, chroma processing, and in-loop filtering.**

Codec-Cooker is a native C/C++ desktop application built with **Dear ImGui** and **OpenGL** that visualizes each stage of a modern block-based codec pipeline. Rather than acting as a media player or encoder front-end, the project focuses on exposing the internal decisions made by a codec and allowing them to be explored interactively.

The goal is to provide a platform for codec experimentation where algorithms can be implemented, visualized, benchmarked, and compared without requiring a complete production encoder.

> **Codec-Cooker is not intended to be another encoder—it is intended to be an research environment for building, visualizing, and evaluating compression algorithms.**

---

# Features

## Image & Multi-Frame Processing

* Load single images or image sequences
* Multi-frame processing pipeline
* Chain analysis filters across consecutive frames
* Foundation for temporal prediction and motion analysis
* Native desktop interface using Dear ImGui

---

## Transform Coding

* Block-based DCT transform
* Configurable block sizes
* Per-block coefficient inspection
* Coefficient heatmaps
* Transform visualization

---

## Quantization

* Uniform quantization
* Dead-zone quantization
* Experimental exponential quantization
* Trellis optimization experiments
* Rate-Distortion Optimized Quantization (RDOQ) experimentation
* Per-block distortion statistics
* Quantization histograms
* Rate-distortion analysis

---

## Color Processing

* Full RGB → YCbCr conversion
* Independent Y, Cb, and Cr channel processing
* Chroma-specific quantization parameters
* Chroma QP offsets
* Per-channel PSNR measurements
* Full color reconstruction (no grayscale approximation)

---

## Entropy Analysis

* Zig-zag coefficient scanning
* Run-length encoding (RLE)
* Laplacian probability modelling
* Context-adaptive probability estimation
* Adaptive entropy experiments
* Estimated bit-cost visualization
* Coefficient probability heatmaps

---

## In-Loop Filtering

Multiple in-loop filtering algorithms are available for experimentation:

* H.264-inspired deblocking filter
* Directional CDEF-style filtering
* Bilateral filtering
* Wiener filtering

The filtering framework exposes:

* Boundary strength calculations
* Flatness detection
* Adaptive filter strength
* Reconstruction quality measurements (PSNR)

allowing direct comparison of reconstruction quality between different filtering techniques.

---

## Analysis & Visualization

* Original vs. reconstructed image comparison
* Block visualization
* Slice visualization
* Heatmaps
* Graph views
* Boundary strength visualization
* PSNR measurements
* Entropy statistics
* Quantization statistics

---

# Research Focus

Codec-Cooker is designed as a research platform for evaluating compression algorithms rather than reproducing a specific codec implementation.

Current areas of experimentation include:

* Transform coding
* Quantization research
* Entropy modelling
* Rate-distortion optimization
* In-loop filtering
* Chroma processing
* Block artifact reduction
* Multi-frame analysis

---

# Architecture

The project is intentionally modular. Major subsystems are implemented independently so they can be mixed, replaced, or extended without affecting the remainder of the pipeline.

```text
Image Input
      │
      ▼
Color Conversion (YCbCr)
      │
      ▼
Block Partitioning
      │
      ▼
Discrete Cosine Transform (DCT)
      │
      ▼
Quantization
      │
      ▼
Entropy Analysis
      │
      ▼
Reconstruction
      │
      ▼
Loop Filtering
      │
      ▼
Visualization & Metrics
```

Each stage is designed to expose intermediate data, making it possible to inspect, compare, and benchmark individual algorithms without requiring a complete codec implementation.

---



# Future Direction

Codec-Cooker is evolving from a still-image compression research tool into a platform for video codec experimentation.

Current development is focused on:

* Temporal prediction
* Motion estimation
* Motion compensation
* Frame referencing
* Bitstream generation
* Video container integration
* Full rate-distortion analysis across frame sequences

Multi-frame support has already been integrated, enabling image sequences to be processed through chained analysis pipelines as a foundation for future temporal coding experiments.

A companion project includes a lightweight sequential MP4 demuxer written in C99, which is planned to become the media input stage for Codec-Cooker's future video analysis pipeline.

---

# Why Codec-Cooker?

Modern codecs such as AV1, AV2, HEVC, and VVC contain hundreds of interacting algorithms. Understanding how those algorithms influence compression efficiency often requires reading hundreds of thousands of lines of production encoder code or stepping through complex encoding pipelines.

Codec-Cooker aims to make those algorithms observable by allowing researchers and developers to isolate individual components, visualize their behavior, compare competing approaches, and rapidly prototype new ideas in an interactive environment.

Rather than being tied to any single codec specification, the project serves as an experimental workbench where modern compression techniques can be explored, analyzed, and extended.





# Video Demo


https://github.com/user-attachments/assets/8977a3b1-4cce-48e4-8857-4101be5a880f


