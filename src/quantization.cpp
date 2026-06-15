#include "quantization.h"
#include <algorithm>
#include <cmath>
#define PI 3.14159265358979323846f

// -------------------------------------------------------
// Build per-coefficient step size matrix
// -------------------------------------------------------
std::vector<float> BuildQuantMatrix(QuantMode mode, int blockSize, float baseQ,
                                    float deadzoneScale) {
  int nn = blockSize * blockSize;
  std::vector<float> mat(nn);
  switch (mode) {
  case QUANT_FLAT:
    for (int i = 0; i < nn; i++)
      mat[i] = baseQ;
    break;
  case QUANT_JPEG:
    if (blockSize == 8) {
      float scale = baseQ / 16.0f;
      for (int i = 0; i < 64; i++)
        mat[i] = std::max(1.0f, JPEG_LUMA[i] * scale);
    } else {
      for (int r = 0; r < blockSize; r++)
        for (int c = 0; c < blockSize; c++) {
          float dist = sqrtf((float)(r * r + c * c)) /
                       sqrtf(2.0f * (blockSize - 1) * (blockSize - 1));
          mat[r * blockSize + c] = std::max(1.0f, baseQ * (1.0f + 4.0f * dist));
        }
    }
    break;
  case QUANT_RAMP:
    for (int r = 0; r < blockSize; r++)
      for (int c = 0; c < blockSize; c++) {
        float t = (float)(r + c) / (float)(2 * (blockSize - 1));
        mat[r * blockSize + c] = std::max(1.0f, baseQ * (1.0f + 3.0f * t));
      }
    break;
  case QUANT_DEADZONE:
  case QUANT_CUSTOM:
    for (int i = 0; i < nn; i++)
      mat[i] = baseQ;
    break;
  }
  return mat;
}

// -------------------------------------------------------
// IDCT-2D (separable)
// -------------------------------------------------------
static float C_dct(int u) { return (u == 0) ? (1.0f / sqrtf(2.0f)) : 1.0f; }

static std::vector<float> IDCT2D(const std::vector<float> &coeffs, int N) {
  std::vector<float> out(N * N, 0.0f);
  for (int y = 0; y < N; y++) {
    for (int x = 0; x < N; x++) {
      float sum = 0.0f;
      for (int u = 0; u < N; u++)
        for (int v = 0; v < N; v++) {
          sum += C_dct(u) * C_dct(v) * coeffs[v * N + u] *
                 cosf((2 * x + 1) * u * PI / (2 * N)) *
                 cosf((2 * y + 1) * v * PI / (2 * N));
        }
      out[y * N + x] = 0.25f * sum + 128.0f;
    }
  }
  return out;
}

// -------------------------------------------------------
// SIGN HIDING (H.265/AV1 technique)
//
// The parity of the sum of all nonzero coefficient signs
// in a block can be used to hide the sign of the LAST
// nonzero coefficient at no rate cost, saving 1 bit/block.
//
// Implementation:
//   1. Compute sign_sum = sum of signs of all nonzero levels.
//   2. If sign_sum parity doesn't match what we need (0),
//      flip the sign of the LAST nonzero coefficient.
//      This costs 0 extra bits — the decoder recovers it
//      from the parity of the others.
//
// Note: this introduces a small distortion on the last
// nonzero coeff but saves a sign bit in the bitstream.
// -------------------------------------------------------
static void ApplySignHiding(std::vector<int> &levels) {
  // Find first and last nonzero positions
  int first_nz = -1, last_nz = -1;
  for (int i = 0; i < (int)levels.size(); i++) {
    if (levels[i] != 0) {
      if (first_nz < 0)
        first_nz = i;
      last_nz = i;
    }
  }
  // Need at least 2 nonzero coefficients to hide a sign
  if (first_nz < 0 || first_nz == last_nz)
    return;

  // Compute sign sum of all nonzero coefficients
  int sign_sum = 0;
  for (int i = first_nz; i <= last_nz; i++)
    if (levels[i] != 0)
      sign_sum += (levels[i] > 0 ? 1 : -1);

  // If parity is odd, flip last nonzero sign to make it even.
  // (Parity convention: even sign_sum = 0 mod 2 → sign hidden as 0)
  if ((sign_sum & 1) != 0) {
    // Flip sign of last nonzero (small distortion, 1 bit saved)
    levels[last_nz] = -levels[last_nz];
  }
}

// -------------------------------------------------------
// Quantize a block — full per-coefficient breakdown
// -------------------------------------------------------
QuantResult QuantizeBlock(const DCTBlock &block, int blockSize, QuantMode mode,
                          float baseQ, float deadzoneScale, bool useTrellis,
                          float lambda) {
  int nn = blockSize * blockSize;
  std::vector<float> mat =
      BuildQuantMatrix(mode, blockSize, baseQ, deadzoneScale);

  QuantResult result;
  result.blockIndex = block.bx * 10000 + block.by;
  result.blockSize = blockSize;
  result.entries.resize(nn);
  result.nonzeroCount = 0;
  result.estimatedBits = 0.0;

  std::vector<float> dequantCoeffs(nn);
  std::vector<int> levels(nn);

  for (int i = 0; i < nn; i++) {
    float val = block.coeffs[i];
    float step = mat[i];
    int level = 0;

    if (useTrellis) {
      int lf = (int)(val / step);
      int lc = lf + (val >= 0 ? 1 : -1);
      float rf = (float)lf * step, rc = (float)lc * step;
      float d0 = (val - rf) * (val - rf), d1 = (val - rc) * (val - rc);
      float r0 = (float)abs(lf), r1 = (float)abs(lc);
      level = (d0 + lambda * r0 <= d1 + lambda * r1) ? lf : lc;
    } else if (mode == QUANT_DEADZONE) {
      float half = step * 0.5f * deadzoneScale;
      if (val > -half && val < half)
        level = 0;
      else
        level = (int)(val / step + (val >= 0 ? 0.5f : -0.5f));
    } else {
      level = (int)(val / step + (val >= 0 ? 0.5f : -0.5f));
    }
    levels[i] = level;
  }

  // Sign hiding is applied to the level array BEFORE dequant
  // (caller must set sign_hiding=true via the block's sign_hiding flag;
  //  here we expose it as a parameter via the QuantResult struct —
  //  the actual call is in main.cpp which passes use_sign_hiding)
  // Sign hiding is applied by caller after this function via
  // ApplySignHidingToResult.

  for (int i = 0; i < nn; i++) {
    float val = block.coeffs[i];
    float step = mat[i];
    int level = levels[i];
    float recon = (float)level * step;
    float error = val - recon;

    result.entries[i] = {val, step, level, recon, error};
    dequantCoeffs[i] = recon;
    if (level != 0) {
      result.nonzeroCount++;
      result.estimatedBits += log2((double)(abs(level) + 1)) + 1.0;
    }
  }

  result.reconPixels = IDCT2D(dequantCoeffs, blockSize);
  return result;
}

// -------------------------------------------------------
// Apply sign hiding to an already-quantized result.
// Call this from main.cpp after QuantizeBlock() when
// use_sign_hiding is enabled.
// Updates levels + reconstructed values + estimatedBits.
// -------------------------------------------------------
void ApplySignHidingToResult(QuantResult &res) {
  int nn = res.blockSize * res.blockSize;
  std::vector<int> levels(nn);
  for (int i = 0; i < nn; i++)
    levels[i] = res.entries[i].level;

  // Find first and last nonzero
  int first_nz = -1, last_nz = -1;
  for (int i = 0; i < nn; i++)
    if (levels[i] != 0) {
      if (first_nz < 0)
        first_nz = i;
      last_nz = i;
    }
  if (first_nz < 0 || first_nz == last_nz)
    return;

  int sign_sum = 0;
  for (int i = first_nz; i <= last_nz; i++)
    if (levels[i] != 0)
      sign_sum += (levels[i] > 0 ? 1 : -1);

  if ((sign_sum & 1) != 0) {
    // Flip last nonzero sign
    levels[last_nz] = -levels[last_nz];
    // Update that entry
    float step = res.entries[last_nz].step;
    float orig = res.entries[last_nz].original;
    float recon = (float)levels[last_nz] * step;
    res.entries[last_nz].level = levels[last_nz];
    res.entries[last_nz].reconstructed = recon;
    res.entries[last_nz].error = orig - recon;
    // Save 1 sign bit
    res.estimatedBits -= 1.0;
  }

  // Rebuild reconPixels with updated levels
  int N = res.blockSize;
  std::vector<float> dequant(nn);
  for (int i = 0; i < nn; i++)
    dequant[i] = (float)levels[i] * res.entries[i].step;
  std::vector<float> out(N * N, 0.0f);
  for (int y = 0; y < N; y++)
    for (int x = 0; x < N; x++) {
      float sum = 0;
      for (int u = 0; u < N; u++)
        for (int v = 0; v < N; v++) {
          float cu = (u == 0) ? 1.0f / sqrtf(2.0f) : 1.0f;
          float cv = (v == 0) ? 1.0f / sqrtf(2.0f) : 1.0f;
          sum += cu * cv * dequant[v * N + u] *
                 cosf((2 * x + 1) * u * PI / (2 * N)) *
                 cosf((2 * y + 1) * v * PI / (2 * N));
        }
      out[y * N + x] = 0.25f * sum + 128.0f;
    }
  res.reconPixels = out;
}

// -------------------------------------------------------
// Reconstruct pixels from dequantized coefficients
// -------------------------------------------------------
std::vector<float> ReconstructPixels(const std::vector<float> &dequantCoeffs,
                                     int blockSize) {
  return IDCT2D(dequantCoeffs, blockSize);
}

// -------------------------------------------------------
// PSNR
// -------------------------------------------------------
float ComputePSNR(const std::vector<unsigned char> &orig,
                  const std::vector<float> &recon, int bx, int by,
                  int blockSize, int imgWidth, int imgHeight) {
  double mse = 0;
  int count = 0;
  for (int r = 0; r < blockSize; r++)
    for (int c = 0; c < blockSize; c++) {
      int ix = bx * blockSize + c, iy = by * blockSize + r;
      if (ix >= imgWidth || iy >= imgHeight)
        continue;
      float o = (float)orig[iy * imgWidth + ix];
      float re = recon[r * blockSize + c];
      if (re < 0)
        re = 0;
      if (re > 255)
        re = 255;
      double d = o - re;
      mse += d * d;
      count++;
    }
  if (count == 0 || mse < 1e-10)
    return 99.99f;
  mse /= count;
  return (float)(10.0 * log10(255.0 * 255.0 / mse));
}
