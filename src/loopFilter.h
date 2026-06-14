#pragma once
#include <cmath>
#include <vector>

enum LoopFilterMode {
  LF_NONE = 0,
  LF_DEBLOCK = 1,   // H.264-style threshold deblocking
  LF_ADAPTIVE = 2,  // gradient-adaptive (CDEF-lite)
  LF_BILATERAL = 3, // edge-preserving bilateral smooth
  LF_WIENER = 4,    // simplified Wiener restoration
};

struct BoundaryStrength {
  int bx, by;      // block grid coords of left/top block of the edge
  bool horizontal; // true = horizontal edge (top block | bottom block)
  float strength;  // 0..1
};

struct LoopFilterResult {
  int width, height;
  std::vector<float> filtered;
  std::vector<BoundaryStrength> boundaries;
  float avg_boundary_strength;
  float psnr_before;
  float psnr_after;
};

struct LoopFilterParams {
  LoopFilterMode mode = LF_DEBLOCK;
  float strength = 1.0f;
  float threshold = 16.0f;
  float flat_thresh = 8.0f;
  float bilateral_sigma_space = 2.0f;
  float bilateral_sigma_color = 20.0f;
  int wiener_radius = 3;
};

std::vector<BoundaryStrength>
ComputeBoundaryStrengths(const std::vector<float> &pixels, int width,
                         int height, int blockSize);

LoopFilterResult ApplyLoopFilter(const std::vector<float> &recon_pixels,
                                 int width, int height, int blockSize,
                                 const LoopFilterParams &params);

float LFPSNR(const std::vector<unsigned char> &orig,
             const std::vector<float> &recon, int width, int height);
