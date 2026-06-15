#include "chain.h"
#include "loopFilter.h"
#include <cmath>

// -------------------------------------------------------
// PSNR helper (float buffer vs uchar reference)
// -------------------------------------------------------
static float ChainPSNR(const std::vector<unsigned char> &orig,
                       const std::vector<float> &recon, int width, int height) {
  return LFPSNR(orig, recon, width, height);
}

// -------------------------------------------------------
// RunChain
// Applies each enabled step sequentially.
// Stage 0 = input baseline (no filter).
// Stages 1..N = one result per enabled ChainStep.
// -------------------------------------------------------
std::vector<ChainStageResult>
RunChain(const std::vector<float> &input_pixels,
         const std::vector<unsigned char> &orig_gray, int width, int height,
         int blockSize, const std::vector<ChainStep> &steps) {
  std::vector<ChainStageResult> results;

  // Stage 0: baseline input
  ChainStageResult baseline;
  baseline.label = "Input (pre-chain)";
  baseline.pixels = input_pixels;
  baseline.psnr = ChainPSNR(orig_gray, input_pixels, width, height);
  baseline.psnr_gain = 0.0f;
  results.push_back(baseline);

  // Run each enabled step
  for (auto &step : steps) {
    if (!step.enabled)
      continue;

    // Get current pixel buffer (output of last stage)
    const std::vector<float> &prev = results.back().pixels;

    // Apply the filter for this step
    LoopFilterParams params = step.ToParams();
    LoopFilterResult lf =
        ApplyLoopFilter(prev, width, height, blockSize, params);

    ChainStageResult stage;
    stage.label = step.Label();
    stage.pixels = lf.filtered;
    stage.psnr = ChainPSNR(orig_gray, lf.filtered, width, height);
    stage.psnr_gain = stage.psnr - results.back().psnr;
    results.push_back(stage);
  }

  return results;
}
