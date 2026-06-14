#include "loopFilter.h"
#include <algorithm>
#include <cmath>
#include <cstring>

static inline float Clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// -------------------------------------------------------
// PSNR utility
// -------------------------------------------------------
float LFPSNR(const std::vector<unsigned char> &orig,
             const std::vector<float> &recon, int width, int height) {
  double mse = 0;
  int n = width * height;
  for (int i = 0; i < n; i++) {
    double d = (double)orig[i] - Clampf(recon[i], 0, 255);
    mse += d * d;
  }
  mse /= n;
  if (mse < 1e-10)
    return 99.99f;
  return (float)(10.0 * log10(255.0 * 255.0 / mse));
}

// -------------------------------------------------------
// Compute boundary strength map
// High strength = big pixel difference across block edge
// -------------------------------------------------------
std::vector<BoundaryStrength>
ComputeBoundaryStrengths(const std::vector<float> &pixels, int width,
                         int height, int blockSize) {

  std::vector<BoundaryStrength> bs;
  int bw = (width + blockSize - 1) / blockSize;
  int bh = (height + blockSize - 1) / blockSize;

  auto pix = [&](int x, int y) -> float {
    if (x < 0 || x >= width || y < 0 || y >= height)
      return 128.0f;
    return pixels[y * width + x];
  };

  // Horizontal edges (between block row by and by+1)
  for (int by = 0; by < bh - 1; by++) {
    for (int bx = 0; bx < bw; bx++) {
      int edge_y = (by + 1) * blockSize;
      float diff = 0;
      for (int c = 0; c < blockSize; c++) {
        int x = bx * blockSize + c;
        diff += fabsf(pix(x, edge_y) - pix(x, edge_y - 1));
      }
      diff /= blockSize;
      bs.push_back({bx, by, true, Clampf(diff / 64.0f, 0, 1)});
    }
  }

  // Vertical edges (between block col bx and bx+1)
  for (int by = 0; by < bh; by++) {
    for (int bx = 0; bx < bw - 1; bx++) {
      int edge_x = (bx + 1) * blockSize;
      float diff = 0;
      for (int r = 0; r < blockSize; r++) {
        int y = by * blockSize + r;
        diff += fabsf(pix(edge_x, y) - pix(edge_x - 1, y));
      }
      diff /= blockSize;
      bs.push_back({bx, by, false, Clampf(diff / 64.0f, 0, 1)});
    }
  }

  return bs;
}

// -------------------------------------------------------
// H.264-STYLE DEBLOCKING FILTER
// Applied along each block boundary, 4 pixels deep each side
// -------------------------------------------------------
static void DeblockEdge(std::vector<float> &pixels, int width, int height,
                        bool horizontal, int edge, int span, float threshold,
                        float flat_thresh, float strength) {
  int len = horizontal ? width : height;

  for (int i = 0; i < len; i++) {
    // p3 p2 p1 p0 | q0 q1 q2 q3   (| = block boundary)
    int px = horizontal ? i : edge;
    int py = horizontal ? edge : i;

    auto get = [&](int dx, int dy) -> float {
      int x = px + dx, y = py + dy;
      if (x < 0 || x >= width || y < 0 || y >= height)
        return 128;
      return pixels[y * width + x];
    };
    auto set = [&](int dx, int dy, float v) {
      int x = px + dx, y = py + dy;
      if (x >= 0 && x < width && y >= 0 && y < height)
        pixels[y * width + x] = Clampf(v, 0, 255);
    };

    int dx0 = horizontal ? 0 : -1;
    int dy0 = horizontal ? -1 : 0;
    int dx1 = horizontal ? 0 : 0;
    int dy1 = horizontal ? 0 : 0;

    float p0 = get(dx0 * 1, dy0 * 1);
    float p1 = get(dx0 * 2, dy0 * 2);
    float p2 = get(dx0 * 3, dy0 * 3);
    float q0 = get(dx1, dy1);
    float q1 = get(dx1 + (horizontal ? 0 : 1), dy1 + (horizontal ? 1 : 0));
    float q2 = get(dx1 + (horizontal ? 0 : 2), dy1 + (horizontal ? 2 : 0));

    float diff = fabsf(p0 - q0);
    if (diff > threshold * strength)
      continue; // strong edge, don't filter

    // Flatness check
    bool flat = (fabsf(p1 - p0) < flat_thresh) &&
                (fabsf(q1 - q0) < flat_thresh) &&
                (fabsf(p2 - p0) < flat_thresh * 2) &&
                (fabsf(q2 - q0) < flat_thresh * 2);

    if (flat) {
      // Strong filter (smooth ramp across boundary)
      float avg = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2) / 9.0f;
      float np0 = (p2 + 2 * p1 + 2 * p0 + q0 + 2) / 6.0f;
      float np1 = (p2 + p1 + p0 + q0 + 2) / 4.0f;
      float nq0 = (p0 + 2 * q0 + 2 * q1 + q2 + 2) / 6.0f;
      float nq1 = (p0 + q0 + q1 + q2 + 2) / 4.0f;
      set(dx0 * 1, dy0 * 1, Clampf(np0, 0, 255));
      set(dx0 * 2, dy0 * 2, Clampf(np1, 0, 255));
      set(dx1, dy1, Clampf(nq0, 0, 255));
      set(dx1 + (horizontal ? 0 : 1), dy1 + (horizontal ? 1 : 0),
          Clampf(nq1, 0, 255));
    } else {
      // Weak filter — only blend p0/q0
      float delta =
          Clampf((q0 - p0) / 4.0f, -threshold / 2, threshold / 2) * strength;
      set(dx0 * 1, dy0 * 1, Clampf(p0 + delta, 0, 255));
      set(dx1, dy1, Clampf(q0 - delta, 0, 255));
    }
  }
}

// -------------------------------------------------------
// GRADIENT-ADAPTIVE FILTER (CDEF-lite)
// Applies a directional weighted median along dominant gradient
// -------------------------------------------------------
static void ApplyCDEFLite(std::vector<float> &pixels, int width, int height,
                          float strength) {
  std::vector<float> output = pixels;
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      float c = pixels[y * width + x];
      // 8 neighbors
      float n[8] = {
          pixels[(y - 1) * width + (x - 1)], pixels[(y - 1) * width + x],
          pixels[(y - 1) * width + (x + 1)], pixels[y * width + (x - 1)],
          pixels[y * width + (x + 1)],       pixels[(y + 1) * width + (x - 1)],
          pixels[(y + 1) * width + x],       pixels[(y + 1) * width + (x + 1)]};
      // Directional sums: H, V, D1, D2
      float h = fabsf(n[3] - c) + fabsf(n[4] - c);
      float v = fabsf(n[1] - c) + fabsf(n[6] - c);
      float d1 = fabsf(n[0] - c) + fabsf(n[7] - c);
      float d2 = fabsf(n[2] - c) + fabsf(n[5] - c);

      float min_dir = std::min({h, v, d1, d2});
      float filtered;
      if (min_dir == h)
        filtered = (n[3] + c + n[4]) / 3.0f;
      else if (min_dir == v)
        filtered = (n[1] + c + n[6]) / 3.0f;
      else if (min_dir == d1)
        filtered = (n[0] + c + n[7]) / 3.0f;
      else
        filtered = (n[2] + c + n[5]) / 3.0f;

      output[y * width + x] = Clampf(c + (filtered - c) * strength, 0, 255);
    }
  }
  pixels = output;
}

// -------------------------------------------------------
// BILATERAL FILTER
// -------------------------------------------------------
static void ApplyBilateral(std::vector<float> &pixels, int width, int height,
                           float sigma_space, float sigma_color,
                           float strength) {
  std::vector<float> output = pixels;
  int radius = (int)(sigma_space * 2 + 0.5f);
  if (radius < 1)
    radius = 1;
  float ss2 = 2 * sigma_space * sigma_space;
  float sc2 = 2 * sigma_color * sigma_color;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float center = pixels[y * width + x];
      float sum = 0, wsum = 0;
      for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
          int nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height)
            continue;
          float neighbor = pixels[ny * width + nx];
          float ws = expf(-(dx * dx + dy * dy) / ss2);
          float wc = expf(-(center - neighbor) * (center - neighbor) / sc2);
          float w = ws * wc;
          sum += neighbor * w;
          wsum += w;
        }
      }
      float filtered = wsum > 1e-6f ? sum / wsum : center;
      output[y * width + x] =
          Clampf(center + (filtered - center) * strength, 0, 255);
    }
  }
  pixels = output;
}

// -------------------------------------------------------
// WIENER FILTER (separable, simplified)
// Estimates signal from a local window, suppresses noise
// -------------------------------------------------------
static void ApplyWiener(std::vector<float> &pixels, int width, int height,
                        int radius, float strength) {
  std::vector<float> output = pixels;
  float noise_var = 100.0f; // assumed noise variance

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float sum = 0, sum2 = 0;
      int count = 0;
      for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
          int nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height)
            continue;
          float v = pixels[ny * width + nx];
          sum += v;
          sum2 += v * v;
          count++;
        }
      }
      float mean = sum / count;
      float var = sum2 / count - mean * mean;
      float local_var = var > noise_var ? var - noise_var : 0;
      float denom = var < 1e-6f ? 1.0f : var;
      float filtered =
          mean + (local_var / denom) * (pixels[y * width + x] - mean);
      output[y * width + x] = Clampf(
          pixels[y * width + x] + (filtered - pixels[y * width + x]) * strength,
          0, 255);
    }
  }
  pixels = output;
}

// -------------------------------------------------------
// MAIN ENTRY POINT
// -------------------------------------------------------
LoopFilterResult ApplyLoopFilter(const std::vector<float> &recon_pixels,
                                 int width, int height, int blockSize,
                                 const LoopFilterParams &params) {

  LoopFilterResult result;
  result.width = width;
  result.height = height;
  result.filtered = recon_pixels; // start with copy

  // Compute boundary strengths before filtering
  result.boundaries =
      ComputeBoundaryStrengths(result.filtered, width, height, blockSize);

  float sum_strength = 0;
  for (auto &b : result.boundaries)
    sum_strength += b.strength;
  result.avg_boundary_strength =
      result.boundaries.empty() ? 0 : sum_strength / result.boundaries.size();

  if (params.mode == LF_NONE)
    return result;

  if (params.mode == LF_DEBLOCK) {
    // Apply deblocking at every block boundary
    int bw = (width + blockSize - 1) / blockSize;
    int bh = (height + blockSize - 1) / blockSize;

    // Horizontal edges
    for (int by = 1; by < bh; by++) {
      int edge_y = by * blockSize;
      if (edge_y < height)
        DeblockEdge(result.filtered, width, height, true, edge_y, blockSize,
                    params.threshold, params.flat_thresh, params.strength);
    }
    // Vertical edges
    for (int bx = 1; bx < bw; bx++) {
      int edge_x = bx * blockSize;
      if (edge_x < width)
        DeblockEdge(result.filtered, width, height, false, edge_x, blockSize,
                    params.threshold, params.flat_thresh, params.strength);
    }
  } else if (params.mode == LF_ADAPTIVE) {
    ApplyCDEFLite(result.filtered, width, height, params.strength);
  } else if (params.mode == LF_BILATERAL) {
    ApplyBilateral(result.filtered, width, height, params.bilateral_sigma_space,
                   params.bilateral_sigma_color, params.strength);
  } else if (params.mode == LF_WIENER) {
    ApplyWiener(result.filtered, width, height, params.wiener_radius,
                params.strength);
  }

  return result;
}
