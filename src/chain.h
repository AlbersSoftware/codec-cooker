#pragma once
#include "loopFilter.h"
#include <string>
#include <vector>

// =====================================================
// CHAIN — pipeline of sequential loop filter stages
// Each step takes the output of the previous as input.
// Modes remain separate (use Loop Filter tab for single
// filter analysis). Chain tab is for multi-stage pipelines.
// =====================================================

enum ChainStepType {
  CHAIN_DEBLOCK = 0,   // H.264/AV1-style deblocking
  CHAIN_CDEF = 1,      // Constrained Directional Enhancement
  CHAIN_BILATERAL = 2, // Edge-preserving bilateral smooth
  CHAIN_WIENER = 3,    // Wiener restoration filter
};

// -------------------------------------------------------
// One step in the chain — wraps params for a single filter
// -------------------------------------------------------
struct ChainStep {
  ChainStepType type = CHAIN_DEBLOCK;
  bool enabled = true;

  // Per-step tuning params
  float strength = 1.0f;
  float threshold = 16.0f;             // deblock only
  float flat_thresh = 8.0f;            // deblock only
  float bilateral_sigma_space = 2.0f;  // bilateral only
  float bilateral_sigma_color = 20.0f; // bilateral only
  int wiener_radius = 3;               // wiener only

  // Human-readable label for UI
  const char *Label() const {
    static const char *names[] = {"Deblock", "CDEF", "Bilateral", "Wiener"};
    return names[(int)type];
  }

  // Convert to LoopFilterParams for ApplyLoopFilter
  LoopFilterParams ToParams() const {
    LoopFilterParams p;
    p.strength = strength;
    p.threshold = threshold;
    p.flat_thresh = flat_thresh;
    p.bilateral_sigma_space = bilateral_sigma_space;
    p.bilateral_sigma_color = bilateral_sigma_color;
    p.wiener_radius = wiener_radius;
    switch (type) {
    case CHAIN_DEBLOCK:
      p.mode = LF_DEBLOCK;
      break;
    case CHAIN_CDEF:
      p.mode = LF_ADAPTIVE;
      break;
    case CHAIN_BILATERAL:
      p.mode = LF_BILATERAL;
      break;
    case CHAIN_WIENER:
      p.mode = LF_WIENER;
      break;
    }
    return p;
  }
};

// -------------------------------------------------------
// Result for one stage in the chain
// -------------------------------------------------------
struct ChainStageResult {
  std::string label;
  std::vector<float> pixels; // pixel buffer AFTER this stage
  float psnr;                // vs original gray
  float psnr_gain;           // delta vs previous stage
};

// -------------------------------------------------------
// Run the full chain on an input float pixel buffer.
// orig_gray: reference image for PSNR.
// Returns one ChainStageResult per enabled step,
// plus index 0 = input (pre-chain) baseline.
// -------------------------------------------------------
std::vector<ChainStageResult>
RunChain(const std::vector<float> &input_pixels,
         const std::vector<unsigned char> &orig_gray, int width, int height,
         int blockSize, const std::vector<ChainStep> &steps);
