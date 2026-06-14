#include "entropy.h"
#include <algorithm>
#include <climits>
#include <cmath>

// -------------------------------------------------------
// Zigzag scan order for NxN block
// Returns flat indices visited in zigzag (DC first)
// -------------------------------------------------------
std::vector<int> ZigzagOrder(int N) {
  std::vector<int> order;
  order.reserve(N * N);
  int r = 0, c = 0;
  bool going_up = true;
  while ((int)order.size() < N * N) {
    order.push_back(r * N + c);
    if (going_up) {
      if (c == N - 1) {
        r++;
        going_up = false;
      } else if (r == 0) {
        c++;
        going_up = false;
      } else {
        r--;
        c++;
      }
    } else {
      if (r == N - 1) {
        c++;
        going_up = true;
      } else if (c == 0) {
        r++;
        going_up = true;
      } else {
        r++;
        c--;
      }
    }
  }
  return order;
}

// -------------------------------------------------------
// Estimate Laplacian scale parameter sigma from values
// MLE estimate: sigma = mean(|x|)
// -------------------------------------------------------
float EstimateLaplacianSigma(const std::vector<float> &values) {
  if (values.empty())
    return 1.0f;
  double sum = 0;
  for (float v : values)
    sum += fabs(v);
  float sigma = (float)(sum / values.size());
  return sigma < 0.5f ? 0.5f : sigma;
}

// -------------------------------------------------------
// Probability of integer level under Laplacian(0, sigma)
// P(level) = integral from level-0.5 to level+0.5 of
//            (1/(2*sigma)) * exp(-|x|/sigma) dx
// -------------------------------------------------------
float LaplaceProb(int level, float sigma) {
  if (sigma < 0.1f)
    sigma = 0.1f;
  double lo = fabs((double)level - 0.5);
  double hi = fabs((double)level + 0.5);
  // CDF of |Laplacian| = 1 - exp(-x/sigma)
  double p = exp(-lo / sigma) - exp(-hi / sigma);
  // sign doubles probability for nonzero (positive and negative)
  if (level != 0)
    p *= 2.0;
  if (p < 1e-12)
    p = 1e-12;
  return (float)p;
}

// -------------------------------------------------------
// Run-length encode coefficient levels (zigzag order)
// -------------------------------------------------------
std::vector<RLEEntry> RunLengthEncode(const std::vector<int> &levels) {
  std::vector<RLEEntry> rle;
  int run = 0;
  for (int i = 0; i < (int)levels.size(); i++) {
    if (levels[i] == 0) {
      run++;
    } else {
      rle.push_back({run, levels[i]});
      run = 0;
    }
  }
  rle.push_back({0, INT_MIN}); // EOB
  return rle;
}

// -------------------------------------------------------
// Estimate bits for an RLE sequence
// run coded as unary (simple model), level as log2
// -------------------------------------------------------
double RLEBitCost(const std::vector<RLEEntry> &rle) {
  double bits = 0;
  for (auto &e : rle) {
    if (e.level == INT_MIN) {
      bits += 2.0;
      break;
    }                                               // EOB flag
    bits += (double)(e.run + 1);                    // unary run
    bits += log2((double)(abs(e.level) + 1)) + 1.0; // magnitude + sign
  }
  return bits;
}

// -------------------------------------------------------
// Main entropy analysis for one block
// -------------------------------------------------------
EntropyResult AnalyzeEntropy(const QuantResult &qres, EntropyModel model,
                             ContextModel &ctx, float fixed_sigma) {
  int N = qres.blockSize;
  int nn = N * N;

  EntropyResult res;
  res.blockIdx = qres.blockIndex;
  res.blockSize = N;
  res.entries.resize(nn);
  res.position_bit_costs.resize(nn, 0.0f);
  res.total_bits_naive = 0;
  res.total_bits_model = 0;
  res.nonzero_count = 0;

  // Collect original coefficient floats for sigma estimation
  std::vector<float> coeff_vals(nn);
  std::vector<int> levels(nn);
  for (int i = 0; i < nn; i++) {
    coeff_vals[i] = qres.entries[i].original;
    levels[i] = qres.entries[i].level;
  }

  // Estimate global sigma for Laplacian model
  float global_sigma = EstimateLaplacianSigma(coeff_vals);

  // Zigzag order for RLE
  std::vector<int> zigzag = ZigzagOrder(N);
  std::vector<int> zigzag_levels(nn);
  for (int i = 0; i < nn; i++)
    zigzag_levels[i] = levels[zigzag[i]];

  // RLE on zigzag-ordered levels
  res.rle = RunLengthEncode(zigzag_levels);
  res.total_bits_rle = RLEBitCost(res.rle);

  // Per-coefficient bit costs
  for (int i = 0; i < nn; i++) {
    int level = levels[i];
    int r = i / N;
    int c = i % N;
    float sigma = global_sigma;
    float prob = 0.0f;

    // Naive: assume uniform over [-127, 127]
    double naive_bits = 8.0; // 8 bits per coefficient raw
    res.total_bits_naive += naive_bits;

    switch (model) {
    case ENTROPY_FIXED_PROB:
      // flat prob over typical range
      prob = 1.0f / 255.0f;
      break;

    case ENTROPY_LAPLACIAN:
      // Per-position sigma: AC coefficients at high freq have smaller sigma
      {
        float dist = sqrtf((float)(r * r + c * c)) /
                     sqrtf(2.0f * (N - 1) * (N - 1) + 1e-6f);
        sigma = global_sigma * (1.0f - 0.7f * dist);
        if (sigma < 0.5f)
          sigma = 0.5f;
        prob = LaplaceProb(level, sigma);
      }
      break;

    case ENTROPY_CONTEXT:
      prob = ctx.Probability(i, abs(level));
      // sign bit cost if nonzero
      if (level != 0)
        prob *= 0.5f; // halve for sign uncertainty
      if (prob < 1e-12f)
        prob = 1e-12f;
      break;

    case ENTROPY_ADAPTIVE:
      // Use context model but also update it online
      prob = ctx.Probability(i, abs(level));
      if (level != 0)
        prob *= 0.5f;
      if (prob < 1e-12f)
        prob = 1e-12f;
      break;
    }

    float bit_cost = (prob > 1e-12f) ? (float)(-log2((double)prob)) : 20.0f;

    res.entries[i] = {i, r, c, level, prob, bit_cost, sigma};
    res.position_bit_costs[i] = bit_cost;
    res.total_bits_model += bit_cost;

    if (level != 0)
      res.nonzero_count++;
  }

  // Update context model with this block's levels
  ctx.Update(levels);

  return res;
}
