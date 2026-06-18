#include "multiframe.h"
#include "stb_image.h"
#include <algorithm>
#include <cmath>

static float Clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Same BT.601 full-range conversions used in main.cpp, duplicated here
// so multiframe.cpp has no dependency on main.cpp's statics.
static unsigned char RGBtoY(unsigned char r, unsigned char g, unsigned char b) {
  return (unsigned char)Clampf(0.299f * r + 0.587f * g + 0.114f * b, 0, 255);
}
static unsigned char RGBtoCb(unsigned char r, unsigned char g,
                             unsigned char b) {
  return (unsigned char)Clampf(
      -0.168736f * r - 0.331264f * g + 0.5f * b + 128.f, 0, 255);
}
static unsigned char RGBtoCr(unsigned char r, unsigned char g,
                             unsigned char b) {
  return (unsigned char)Clampf(0.5f * r - 0.418688f * g - 0.081312f * b + 128.f,
                               0, 255);
}
static unsigned char YCbCrtoR(float Y, float Cb, float Cr) {
  return (unsigned char)Clampf(Y + 1.402f * (Cr - 128.f), 0, 255);
}
static unsigned char YCbCrtoG(float Y, float Cb, float Cr) {
  return (unsigned char)Clampf(
      Y - 0.344136f * (Cb - 128.f) - 0.714136f * (Cr - 128.f), 0, 255);
}
static unsigned char YCbCrtoB(float Y, float Cb, float Cr) {
  return (unsigned char)Clampf(Y + 1.772f * (Cb - 128.f), 0, 255);
}

static float ChannelPSNR(const std::vector<unsigned char> &orig,
                         const std::vector<float> &recon) {
  if (orig.size() != recon.size())
    return 0;
  double mse = 0;
  for (size_t i = 0; i < orig.size(); i++) {
    double d = (double)orig[i] - Clampf(recon[i], 0, 255);
    mse += d * d;
  }
  mse /= (double)orig.size();
  if (mse < 1e-10)
    return 99.99f;
  return (float)(10.0 * log10(255.0 * 255.0 / mse));
}

static GLuint UploadTex(const unsigned char *rgba, int w, int h) {
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               rgba);
  return tex;
}

static void UpdateTex(GLuint tex, const unsigned char *rgba, int w, int h) {
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE,
                  rgba);
}

void FreeMultiFrameTextures(MultiFrame &frame) {
  if (frame.texOriginal) {
    glDeleteTextures(1, &frame.texOriginal);
    frame.texOriginal = 0;
  }
  if (frame.texRecon) {
    glDeleteTextures(1, &frame.texRecon);
    frame.texRecon = 0;
  }
  if (frame.texChain) {
    glDeleteTextures(1, &frame.texChain);
    frame.texChain = 0;
  }
}

std::vector<MultiFrame> LoadMultiFrames(const std::vector<std::string> &paths,
                                        std::vector<std::string> &skipped) {
  std::vector<MultiFrame> frames;
  int refW = 0, refH = 0;

  for (auto &path : paths) {
    int w, h, n;
    unsigned char *data = stbi_load(path.c_str(), &w, &h, &n, 4);
    if (!data) {
      skipped.push_back(path);
      continue;
    }
    if (frames.empty()) {
      refW = w;
      refH = h;
    } else if (w != refW || h != refH) {
      // Dimension mismatch: skip rather than silently crop/resize, so
      // the user knows exactly which files didn't make it into the
      // batch and why.
      skipped.push_back(path);
      stbi_image_free(data);
      continue;
    }

    MultiFrame f;
    f.path = path;
    f.width = w;
    f.height = h;
    int npx = w * h;
    f.rgbaOrig.assign(data, data + npx * 4);
    f.Y.resize(npx);
    f.Cb.resize(npx);
    f.Cr.resize(npx);
    for (int i = 0; i < npx; i++) {
      unsigned char r = data[i * 4 + 0], gv = data[i * 4 + 1],
                    b = data[i * 4 + 2];
      f.Y[i] = RGBtoY(r, gv, b);
      f.Cb[i] = RGBtoCb(r, gv, b);
      f.Cr[i] = RGBtoCr(r, gv, b);
    }
    f.texOriginal = UploadTex(data, w, h);
    stbi_image_free(data);
    frames.push_back(std::move(f));
  }
  return frames;
}

// Mirrors QuantizeChannel() in main.cpp exactly, duplicated here so
// multiframe.cpp doesn't depend on main.cpp's file-local statics.
static std::vector<QuantResult>
QuantizeChannelMF(const std::vector<DCTBlock> &blocks,
                  const std::vector<unsigned char> &ref, int width, int height,
                  int blockSize, QuantMode mode, float baseQ,
                  float deadzoneScale, bool useTrellis, float lambda,
                  bool signHiding, std::vector<float> &out_recon) {
  out_recon.assign(width * height, 128.0f);
  std::vector<QuantResult> results;
  for (auto &block : blocks) {
    QuantResult qres = QuantizeBlock(block, blockSize, mode, baseQ,
                                     deadzoneScale, useTrellis, lambda);
    if (signHiding)
      ApplySignHidingToResult(qres);
    qres.psnr = ComputePSNR(ref, qres.reconPixels, block.bx, block.by,
                            blockSize, width, height);
    for (int r = 0; r < blockSize; r++)
      for (int c = 0; c < blockSize; c++) {
        int ix = block.bx * blockSize + c, iy = block.by * blockSize + r;
        if (ix >= width || iy >= height)
          continue;
        out_recon[iy * width + ix] = qres.reconPixels[r * blockSize + c];
      }
    results.push_back(qres);
  }
  return results;
}

void RunMultiFrameExperiment(std::vector<MultiFrame> &frames, int blockSize,
                             QuantMode mode, float baseQ, float chromaQpOffset,
                             float deadzoneScale, bool useTrellis,
                             float trellisLambda, bool signHiding,
                             ContextModel &ctxY, ContextModel &ctxCb,
                             ContextModel &ctxCr,
                             EntropyModel /*entropyModel*/) {
  float qY = baseQ;
  float qC = baseQ + chromaQpOffset;

  for (auto &f : frames) {
    f.blocksY = ComputeDCTBlocks(f.Y, f.width, f.height, blockSize);
    f.blocksCb = ComputeDCTBlocks(f.Cb, f.width, f.height, blockSize);
    f.blocksCr = ComputeDCTBlocks(f.Cr, f.width, f.height, blockSize);

    f.resultsY = QuantizeChannelMF(f.blocksY, f.Y, f.width, f.height, blockSize,
                                   mode, qY, deadzoneScale, useTrellis,
                                   trellisLambda, signHiding, f.reconY);
    f.resultsCb = QuantizeChannelMF(
        f.blocksCb, f.Cb, f.width, f.height, blockSize, mode, qC, deadzoneScale,
        useTrellis, trellisLambda, signHiding, f.reconCb);
    f.resultsCr = QuantizeChannelMF(
        f.blocksCr, f.Cr, f.width, f.height, blockSize, mode, qC, deadzoneScale,
        useTrellis, trellisLambda, signHiding, f.reconCr);

    // Context model statistics accumulate ACROSS frames (ctxY etc. are
    // not reset per-frame by the caller) — this is the one genuine
    // "multi-frame benefit" requested: later frames' is_zero/level
    // probabilities are informed by earlier frames' coefficient
    // statistics, similar to how a real entropy coder's adaptive
    // model carries state across a GOP instead of restarting cold
    // for every frame.
    for (auto &r : f.resultsY) {
      std::vector<int> lv(r.entries.size());
      for (size_t i = 0; i < r.entries.size(); i++)
        lv[i] = r.entries[i].level;
      ctxY.Update(lv);
    }
    for (auto &r : f.resultsCb) {
      std::vector<int> lv(r.entries.size());
      for (size_t i = 0; i < r.entries.size(); i++)
        lv[i] = r.entries[i].level;
      ctxCb.Update(lv);
    }
    for (auto &r : f.resultsCr) {
      std::vector<int> lv(r.entries.size());
      for (size_t i = 0; i < r.entries.size(); i++)
        lv[i] = r.entries[i].level;
      ctxCr.Update(lv);
    }

    f.psnrY = ChannelPSNR(f.Y, f.reconY);
    f.psnrCb = ChannelPSNR(f.Cb, f.reconCb);
    f.psnrCr = ChannelPSNR(f.Cr, f.reconCr);

    f.reconImage.assign(f.width * f.height * 4, 255);
    double mseRgb = 0;
    for (int i = 0; i < f.width * f.height; i++) {
      float Y = f.reconY[i], Cb = f.reconCb[i], Cr = f.reconCr[i];
      unsigned char R = YCbCrtoR(Y, Cb, Cr), G = YCbCrtoG(Y, Cb, Cr),
                    B = YCbCrtoB(Y, Cb, Cr);
      int idx = i * 4;
      f.reconImage[idx + 0] = R;
      f.reconImage[idx + 1] = G;
      f.reconImage[idx + 2] = B;
      f.reconImage[idx + 3] = 255;
      double dr = f.rgbaOrig[idx + 0] - R, dg = f.rgbaOrig[idx + 1] - G,
             db = f.rgbaOrig[idx + 2] - B;
      mseRgb += (dr * dr + dg * dg + db * db) / 3.0;
    }
    mseRgb /= (double)(f.width * f.height);
    f.psnrRgb = (mseRgb < 1e-10)
                    ? 99.99f
                    : (float)(10.0 * log10(255.0 * 255.0 / mseRgb));

    if (f.texRecon)
      UpdateTex(f.texRecon, f.reconImage.data(), f.width, f.height);
    else
      f.texRecon = UploadTex(f.reconImage.data(), f.width, f.height);

    f.experimentRun = true;
    f.chainRun = false;
  }
}

void RunMultiFrameChain(std::vector<MultiFrame> &frames, int blockSize,
                        const std::vector<ChainStep> &steps) {
  for (auto &f : frames) {
    if (!f.experimentRun || f.reconY.empty())
      continue;

    f.chainResults =
        RunChain(f.reconY, f.Y, f.width, f.height, blockSize, steps);
    if (f.chainResults.empty())
      continue;

    auto &finalStage = f.chainResults.back();
    f.chainImage.assign(f.width * f.height * 4, 255);
    for (int i = 0; i < f.width * f.height; i++) {
      float Y = finalStage.pixels[i], Cb = f.reconCb[i], Cr = f.reconCr[i];
      int idx = i * 4;
      f.chainImage[idx + 0] = YCbCrtoR(Y, Cb, Cr);
      f.chainImage[idx + 1] = YCbCrtoG(Y, Cb, Cr);
      f.chainImage[idx + 2] = YCbCrtoB(Y, Cb, Cr);
      f.chainImage[idx + 3] = 255;
    }
    f.chainPsnrY = finalStage.psnr;

    if (f.texChain)
      UpdateTex(f.texChain, f.chainImage.data(), f.width, f.height);
    else
      f.texChain = UploadTex(f.chainImage.data(), f.width, f.height);

    f.chainRun = true;
  }
}
