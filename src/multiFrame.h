#pragma once
#include "chain.h"
#include "dct.h"
#include "entropy.h"
#include "loopFilter.h"
#include "quantization.h"
#include <string>
#include <vector>

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL3/SDL_opengles2.h>
#else
#include <SDL3/SDL_opengl.h>
#endif

// =====================================================
// MULTI-FRAME — applies the existing single-image YCbCr
// DCT/quant/loop-filter/chain pipeline independently to a
// batch of loaded images ("frames" in name only — there is
// no inter-frame prediction here, each frame is coded as
// its own I-frame; see ExperimentRun() per frame). This
// still gives a real multi-image benefit for: the entropy
// ContextModel (which can be carried across frames so block
// statistics accumulate, similar to how real codecs reuse
// probability state across a GOP), and for visually comparing
// how the chain of loop filters performs across a batch.
// =====================================================

struct MultiFrame {
  std::string path;
  int width = 0, height = 0;

  std::vector<unsigned char> rgbaOrig;
  std::vector<unsigned char> Y, Cb, Cr;

  std::vector<DCTBlock> blocksY, blocksCb, blocksCr;

  std::vector<QuantResult> resultsY, resultsCb, resultsCr;
  std::vector<float> reconY, reconCb, reconCr;
  std::vector<unsigned char> reconImage; // RGBA after I-frame pipeline

  // Chain output (Y channel chained, recombined with Cb/Cr like main.cpp)
  std::vector<ChainStageResult> chainResults;
  std::vector<unsigned char> chainImage; // RGBA after chain

  float psnrY = 0, psnrCb = 0, psnrCr = 0, psnrRgb = 0;
  float chainPsnrY = 0;

  GLuint texOriginal = 0;
  GLuint texRecon = 0;
  GLuint texChain = 0;

  bool experimentRun = false;
  bool chainRun = false;
};

// Loads a batch of image files into MultiFrame entries. Frames whose
// dimensions don't match the first loaded frame are skipped (and their
// path returned in `skipped`) since the pipeline assumes a fixed
// width/height across the batch (same as a real GOP).
std::vector<MultiFrame> LoadMultiFrames(const std::vector<std::string> &paths,
                                        std::vector<std::string> &skipped);

// Frees GL textures owned by a frame (call before discarding/reloading).
void FreeMultiFrameTextures(MultiFrame &frame);

// Runs the existing I-frame style pipeline (per-channel quantize +
// YCbCr recombine) on every frame in the batch independently.
// ctxY/ctxCb/ctxCr are NOT reset between frames — block statistics
// accumulate across the batch, matching how a real encoder's entropy
// model adapts over a sequence rather than restarting per frame.
void RunMultiFrameExperiment(std::vector<MultiFrame> &frames, int blockSize,
                             QuantMode mode, float baseQ, float chromaQpOffset,
                             float deadzoneScale, bool useTrellis,
                             float trellisLambda, bool signHiding,
                             ContextModel &ctxY, ContextModel &ctxCb,
                             ContextModel &ctxCr, EntropyModel entropyModel);

// Runs the existing chain pipeline (Y channel) on every frame's
// reconstructed Y, recombining with that frame's own Cb/Cr.
void RunMultiFrameChain(std::vector<MultiFrame> &frames, int blockSize,
                        const std::vector<ChainStep> &steps);
