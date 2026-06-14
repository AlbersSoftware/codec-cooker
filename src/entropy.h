#pragma once
#include "quantization.h"
#include <climits>
#include <cmath>
#include <string>
#include <vector>

// -------------------------------------------------------
// ENTROPY CODING MODELS
// -------------------------------------------------------
enum EntropyModel {
  ENTROPY_FIXED_PROB = 0, // naive flat distribution
  ENTROPY_LAPLACIAN = 1,  // Laplacian (fits DCT coeffs well)
  ENTROPY_CONTEXT = 2,    // per-position context model (AV1-style)
  ENTROPY_ADAPTIVE = 3,   // online adaptive symbol counting
};

// -------------------------------------------------------
// Run-length encoding entry
// level == INT_MIN signals EOB (end of block)
// -------------------------------------------------------
struct RLEEntry {
  int run;
  int level;
};

// -------------------------------------------------------
// Per-coefficient entropy breakdown
// -------------------------------------------------------
struct EntropyCoeffEntry {
  int idx;
  int row, col;
  int level;
  float probability;
  float bit_cost;
  float laplace_sigma;
};

// -------------------------------------------------------
// Full entropy result for one block
// -------------------------------------------------------
struct EntropyResult {
  int blockIdx;
  int blockSize;
  std::vector<EntropyCoeffEntry> entries;
  std::vector<RLEEntry> rle;
  double total_bits_naive;
  double total_bits_model;
  double total_bits_rle;
  int nonzero_count;
  std::vector<float> position_bit_costs;
};

// -------------------------------------------------------
// Context model — per-position histogram across blocks
// -------------------------------------------------------
struct ContextModel {
  int blockSize;
  static const int MAX_LEVEL = 64;
  std::vector<std::vector<int>> counts; // [position][abs_level]
  int total_blocks;

  void Init(int bs) {
    blockSize = bs;
    int nn = bs * bs;
    counts.assign(nn, std::vector<int>(MAX_LEVEL + 1, 1)); // Laplace smoothing
    total_blocks = 0;
  }

  void Update(const std::vector<int> &levels) {
    for (int i = 0; i < (int)levels.size() && i < (int)counts.size(); i++) {
      int l = abs(levels[i]);
      if (l > MAX_LEVEL)
        l = MAX_LEVEL;
      counts[i][l]++;
    }
    total_blocks++;
  }

  float Probability(int pos, int abs_level) const {
    if (pos >= (int)counts.size())
      return 0.01f;
    int al = abs_level > MAX_LEVEL ? MAX_LEVEL : abs_level;
    int total = 0;
    for (int v : counts[pos])
      total += v;
    return (float)counts[pos][al] / (float)total;
  }
};

// -------------------------------------------------------
// API
// -------------------------------------------------------
float EstimateLaplacianSigma(const std::vector<float> &values);
float LaplaceProb(int level, float sigma);
EntropyResult AnalyzeEntropy(const QuantResult &qres, EntropyModel model,
                             ContextModel &ctx, float fixed_sigma = 20.0f);
std::vector<RLEEntry> RunLengthEncode(const std::vector<int> &levels);
double RLEBitCost(const std::vector<RLEEntry> &rle);
std::vector<int> ZigzagOrder(int N);
