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
      // Radial falloff proxy for other sizes
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
    // Same step as flat, deadzone applied separately during quantization
    for (int i = 0; i < nn; i++)
      mat[i] = baseQ;
    break;

  case QUANT_CUSTOM:
    // Caller is expected to fill mat externally; return flat as default
    for (int i = 0; i < nn; i++)
      mat[i] = baseQ;
    break;
  }

  return mat;
}

// -------------------------------------------------------
// IDCT-2D (separable, row then column)
// -------------------------------------------------------
static float C_dct(int u) { return (u == 0) ? (1.0f / sqrtf(2.0f)) : 1.0f; }

static std::vector<float> IDCT2D(const std::vector<float> &coeffs, int N) {
  std::vector<float> out(N * N, 0.0f);
  for (int y = 0; y < N; y++) {
    for (int x = 0; x < N; x++) {
      float sum = 0.0f;
      for (int u = 0; u < N; u++) {
        for (int v = 0; v < N; v++) {
          float cu = C_dct(u);
          float cv = C_dct(v);
          sum += cu * cv * coeffs[v * N + u] *
                 cosf((2 * x + 1) * u * PI / (2 * N)) *
                 cosf((2 * y + 1) * v * PI / (2 * N));
        }
      }
      out[y * N + x] = 0.25f * sum + 128.0f; // level-shift back
    }
  }
  return out;
}

// -------------------------------------------------------
// Quantize a block — returns full per-coefficient breakdown
// -------------------------------------------------------
QuantResult QuantizeBlock(const DCTBlock &block, int blockSize, QuantMode mode,
                          float baseQ, float deadzoneScale, bool useTrellis,
                          float lambda) {
  int nn = blockSize * blockSize;
  std::vector<float> mat =
      BuildQuantMatrix(mode, blockSize, baseQ, deadzoneScale);

  QuantResult result;
  result.blockIndex = block.bx * 10000 + block.by; // unique key
  result.blockSize = blockSize;
  result.entries.resize(nn);
  result.nonzeroCount = 0;
  result.estimatedBits = 0.0;

  std::vector<float> dequantCoeffs(nn);

  for (int i = 0; i < nn; i++) {
    float val = block.coeffs[i];
    float step = mat[i];

    int level = 0;

    if (useTrellis) {
      // Greedy trellis: pick floor or ceil to minimize D + lambda*R
      int lf = (int)(val / step); // floor toward zero
      int lc = lf + (val >= 0 ? 1 : -1);
      float rf = (float)lf * step;
      float rc = (float)lc * step;
      float d0 = (val - rf) * (val - rf);
      float d1 = (val - rc) * (val - rc);
      float r0 = (float)abs(lf);
      float r1 = (float)abs(lc);
      level = (d0 + lambda * r0 <= d1 + lambda * r1) ? lf : lc;
    } else if (mode == QUANT_DEADZONE) {
      // Expanded deadzone: widen the zero region
      float half = step * 0.5f * deadzoneScale;
      if (val > -half && val < half) {
        level = 0;
      } else {
        level = (int)(val / step + (val >= 0 ? 0.5f : -0.5f));
      }
    } else {
      // Standard nearest-neighbor round
      level = (int)(val / step + (val >= 0 ? 0.5f : -0.5f));
    }

    float recon = (float)level * step;
    float error = val - recon;

    result.entries[i].original = val;
    result.entries[i].step = step;
    result.entries[i].level = level;
    result.entries[i].reconstructed = recon;
    result.entries[i].error = error;

    dequantCoeffs[i] = recon;

    if (level != 0) {
      result.nonzeroCount++;
      result.estimatedBits += log2((double)(abs(level) + 1)) + 1.0;
    }
  }

  // Reconstruct pixels via IDCT
  result.reconPixels = IDCT2D(dequantCoeffs, blockSize);

  return result;
}

// -------------------------------------------------------
// Reconstruct pixels from dequantized coefficients
// -------------------------------------------------------
std::vector<float> ReconstructPixels(const std::vector<float> &dequantCoeffs,
                                     int blockSize) {
  return IDCT2D(dequantCoeffs, blockSize);
}

// -------------------------------------------------------
// PSNR between original gray block and reconstructed floats
// -------------------------------------------------------
float ComputePSNR(const std::vector<unsigned char> &orig,
                  const std::vector<float> &recon, int bx, int by,
                  int blockSize, int imgWidth, int imgHeight) {
  double mse = 0.0;
  int count = 0;
  for (int r = 0; r < blockSize; r++) {
    for (int c = 0; c < blockSize; c++) {
      int ix = bx * blockSize + c;
      int iy = by * blockSize + r;
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
  }
  if (count == 0 || mse < 1e-10)
    return 99.99f;
  mse /= count;
  return (float)(10.0 * log10(255.0 * 255.0 / mse));
}
