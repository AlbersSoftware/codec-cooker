#pragma once
#include "dct.h"
#include <cmath>
#include <vector>

// ===== QUANT MODES =====
enum QuantMode {
  QUANT_FLAT = 0,
  QUANT_JPEG = 1,
  QUANT_RAMP = 2,
  QUANT_DEADZONE = 3,
  QUANT_CUSTOM = 4,
};

// Standard JPEG luma quantization matrix
static const float JPEG_LUMA[64] = {
    16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
    14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
    18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
};

struct QuantCoeffEntry {
  float original;      // raw DCT coefficient
  float step;          // step size used
  int level;           // quantized integer level
  float reconstructed; // dequantized value
  float error;         // original - reconstructed
};

struct QuantResult {
  int blockIndex;
  int blockSize;
  std::vector<QuantCoeffEntry> entries; // one per coefficient
  float psnr;
  double estimatedBits;
  int nonzeroCount;
  std::vector<float> reconPixels; // reconstructed pixel block
};

// Build step-size matrix for given mode, blockSize, baseQ, deadzoneScale
std::vector<float> BuildQuantMatrix(QuantMode mode, int blockSize, float baseQ,
                                    float deadzoneScale = 1.0f);

// Quantize a single DCTBlock and return full per-coefficient breakdown
QuantResult QuantizeBlock(const DCTBlock &block, int blockSize, QuantMode mode,
                          float baseQ, float deadzoneScale, bool useTrellis,
                          float lambda);

// Reconstruct pixel block from quantized coefficients (IDCT)
std::vector<float> ReconstructPixels(const std::vector<float> &dequantCoeffs,
                                     int blockSize);

// Compute PSNR between original gray pixels and reconstructed
float ComputePSNR(const std::vector<unsigned char> &orig,
                  const std::vector<float> &recon, int bx, int by,
                  int blockSize, int imgWidth, int imgHeight);
