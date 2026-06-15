#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl3.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL3/SDL_opengles2.h>
#else
#include <SDL3/SDL_opengl.h>
#endif

#include <commdlg.h>
#include <windows.h>

#include "chain.h"
#include "dct.h"
#include "entropy.h"
#include "loopFilter.h"
#include "quantization.h"

// =====================================================
// GLOBALS — textures, image data, block data
// =====================================================
GLuint g_tex_original = 0;
GLuint g_tex_recon = 0;
GLuint g_tex_filtered = 0;
GLuint g_tex_chain_out = 0;
GLuint g_icon_tex = 0;
int g_icon_w = 0;
int g_icon_h = 0;
int g_tex_w = 0;
int g_tex_h = 0;

// CHANGE 1: Replace g_gray/g_rgba_orig/g_dct_blocks with YCbCr channel buffers
std::vector<unsigned char> g_rgba_orig;

// YCbCr channel buffers (BT.601 full-range, 4:4:4)
std::vector<unsigned char> g_Y;
std::vector<unsigned char> g_Cb;
std::vector<unsigned char> g_Cr;

std::vector<float> g_recon_Y;
std::vector<float> g_recon_Cb;
std::vector<float> g_recon_Cr;

std::vector<DCTBlock> g_dct_blocks_Y;
std::vector<DCTBlock> g_dct_blocks_Cb;
std::vector<DCTBlock> g_dct_blocks_Cr;

// Aliases so all existing code referencing g_dct_blocks / g_gray still compiles
std::vector<DCTBlock> &g_dct_blocks = g_dct_blocks_Y;
std::vector<unsigned char> &g_gray = g_Y;

// Per-channel quant results
std::vector<QuantResult> g_quant_results_Y;
std::vector<QuantResult> g_quant_results_Cb;
std::vector<QuantResult> g_quant_results_Cr;

int g_block_size = 8;
int g_selected_block = -1;

// =====================================================
// EXPERIMENT PARAMETERS
// =====================================================

// -- Quantization --
QuantMode g_quant_mode = QUANT_FLAT;
float g_base_q = 16.0f;
// CHANGE 2: Add chroma QP offset
float g_chroma_qp_offset = 4.0f; // Cb/Cr Q = base_q + offset
float g_deadzone_scale = 1.5f;
bool g_use_trellis = false;
float g_trellis_lambda = 0.5f;

// -- Entropy --
EntropyModel g_entropy_model = ENTROPY_LAPLACIAN;
float g_fixed_sigma = 20.0f;
ContextModel g_ctx_model;

// -- Loop Filter --
LoopFilterParams g_lf_params;
bool g_lf_enabled = false;

// =====================================================
// EXPERIMENT RESULTS
// =====================================================
bool g_experiment_run = false;
std::vector<QuantResult> g_quant_results;
std::vector<EntropyResult> g_entropy_results;
LoopFilterResult g_lf_result;
std::vector<unsigned char> g_recon_image;    // RGBA reconstructed
std::vector<unsigned char> g_filtered_image; // RGBA loop-filtered

// CHANGE 3: Per-channel PSNR
float g_psnr_Y = 0, g_psnr_Cb = 0, g_psnr_Cr = 0, g_psnr_rgb = 0;

// =====================================================
// FEATURE TOGGLES
// =====================================================
bool use_sign_hiding = false;
bool use_ccso = false;
bool use_gdf = false;
bool use_grain = false;
bool use_superblocks = false;

static float grain_strength = 8.0f;
static int grain_seed = 42;

// =====================================================
// CHAIN STATE
// =====================================================
std::vector<ChainStep> g_chain_steps;
std::vector<ChainStageResult> g_chain_results;
bool g_chain_run = false;

// =====================================================
// THEME
// =====================================================
struct Theme {
  ImVec4 primary = ImVec4(0.36f, 0.71f, 0.31f, 1.0f);
  ImVec4 bg = ImVec4(0.08f, 0.08f, 0.08f, 1.0f);
  ImVec4 text = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  ImVec4 separator = ImVec4(0.30f, 0.30f, 0.30f, 1.0f);
  ImVec4 tab_bg = ImVec4(0.12f, 0.12f, 0.12f, 1.0f);
  float font_scale = 1.0f;
  int font_idx = 0;
} g_theme;

void ApplyTheme(ImGuiStyle &s) {
  s.Colors[ImGuiCol_WindowBg] = g_theme.bg;
  s.Colors[ImGuiCol_MenuBarBg] = g_theme.bg;
  s.Colors[ImGuiCol_TitleBg] = g_theme.bg;
  s.Colors[ImGuiCol_TitleBgActive] = g_theme.bg;
  s.Colors[ImGuiCol_ChildBg] = g_theme.bg;
  s.Colors[ImGuiCol_Button] = g_theme.primary;
  s.Colors[ImGuiCol_ButtonHovered] =
      ImVec4(g_theme.primary.x * 1.2f, g_theme.primary.y * 1.2f,
             g_theme.primary.z * 1.2f, 1.0f);
  s.Colors[ImGuiCol_Header] = g_theme.primary;
  s.Colors[ImGuiCol_Tab] = g_theme.tab_bg;
  s.Colors[ImGuiCol_TabActive] = g_theme.primary;
  s.Colors[ImGuiCol_TabHovered] = g_theme.primary;
  s.Colors[ImGuiCol_Text] = g_theme.text;
  s.Colors[ImGuiCol_Separator] = g_theme.separator;
  s.Colors[ImGuiCol_SeparatorActive] = g_theme.primary;
  s.Colors[ImGuiCol_SeparatorHovered] = g_theme.primary;
  s.Colors[ImGuiCol_FrameBg] = ImVec4(
      g_theme.bg.x + 0.06f, g_theme.bg.y + 0.06f, g_theme.bg.z + 0.06f, 1.0f);
  s.Colors[ImGuiCol_SliderGrab] = g_theme.primary;
  s.Colors[ImGuiCol_CheckMark] = g_theme.primary;
}

// =====================================================
// UTILITY HELPERS
// =====================================================
static float Clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

static ImU32 HeatColor(float t, unsigned char alpha = 180) {
  t = t < 0 ? 0 : (t > 1 ? 1 : t);
  float r, g, b;
  if (t < 0.25f) {
    float s = t / 0.25f;
    r = 0;
    g = s;
    b = 1;
  } else if (t < 0.50f) {
    float s = (t - 0.25f) / 0.25f;
    r = 0;
    g = 1;
    b = 1 - s;
  } else if (t < 0.75f) {
    float s = (t - 0.50f) / 0.25f;
    r = s;
    g = 1;
    b = 0;
  } else {
    float s = (t - 0.75f) / 0.25f;
    r = 1;
    g = 1 - s;
    b = 0;
  }
  return IM_COL32((int)(r * 255), (int)(g * 255), (int)(b * 255), alpha);
}

bool OpenFileDialog(char *outPath, size_t maxSize) {
  OPENFILENAMEA ofn;
  ZeroMemory(&ofn, sizeof(ofn));
  ZeroMemory(outPath, maxSize);
  ofn.lStructSize = sizeof(ofn);
  ofn.lpstrFilter = "Image Files\0*.png;*.jpg;*.jpeg;*.bmp\0All Files\0*.*\0";
  ofn.lpstrFile = outPath;
  ofn.nMaxFile = (DWORD)maxSize;
  ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_EXPLORER;
  return GetOpenFileNameA(&ofn);
}

bool SaveFileDialog(char *outPath, size_t maxSize, const char *filter,
                    const char *defExt) {
  OPENFILENAMEA ofn;
  ZeroMemory(&ofn, sizeof(ofn));
  ZeroMemory(outPath, maxSize);
  ofn.lStructSize = sizeof(ofn);
  ofn.lpstrFilter = filter;
  ofn.lpstrFile = outPath;
  ofn.nMaxFile = (DWORD)maxSize;
  ofn.lpstrDefExt = defExt;
  ofn.Flags = OFN_OVERWRITEPROMPT | OFN_EXPLORER;
  return GetSaveFileNameA(&ofn);
}

// =====================================================
// TEXTURE HELPERS
// =====================================================
GLuint UploadTexture(const unsigned char *rgba, int w, int h) {
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               rgba);
  return tex;
}

void UpdateTexture(GLuint tex, const unsigned char *rgba, int w, int h) {
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE,
                  rgba);
}

void DeleteTextureSafe(GLuint &tex) {
  if (tex) {
    glDeleteTextures(1, &tex);
    tex = 0;
  }
}

// =====================================================
// CHANGE 4: YCbCr conversion helpers (BT.601 full-range)
// =====================================================
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

// =====================================================
// CHANGE 5: ClearAll — add new channel clears
// =====================================================
void ClearAll() {
  DeleteTextureSafe(g_tex_original);
  DeleteTextureSafe(g_tex_recon);
  DeleteTextureSafe(g_tex_filtered);
  DeleteTextureSafe(g_tex_chain_out);

  g_rgba_orig.clear();
  g_Y.clear();
  g_Cb.clear();
  g_Cr.clear();
  g_recon_Y.clear();
  g_recon_Cb.clear();
  g_recon_Cr.clear();
  g_dct_blocks_Y.clear();
  g_dct_blocks_Cb.clear();
  g_dct_blocks_Cr.clear();
  g_quant_results.clear();
  g_quant_results_Y.clear();
  g_quant_results_Cb.clear();
  g_quant_results_Cr.clear();
  g_entropy_results.clear();
  g_chain_results.clear();
  g_recon_image.clear();
  g_filtered_image.clear();

  g_tex_w = 0;
  g_tex_h = 0;
  g_selected_block = -1;
  g_experiment_run = false;
  g_chain_run = false;
  g_psnr_Y = g_psnr_Cb = g_psnr_Cr = g_psnr_rgb = 0;
  g_ctx_model.Init(g_block_size);
}

// =====================================================
// FILM GRAIN (unchanged)
// =====================================================
static void ApplyFilmGrain(std::vector<unsigned char> &rgba_image, int width,
                           int height, float strength, int seed) {
  auto lcg = [](unsigned &s) -> float {
    s = s * 1664525u + 1013904223u;
    return (float)(s >> 16) / 65535.0f - 0.5f;
  };
  unsigned state = (unsigned)seed;
  for (int i = 0; i < width * height; i++) {
    float noise = lcg(state) * strength;
    int idx = i * 4;
    rgba_image[idx + 0] =
        (unsigned char)Clampf(rgba_image[idx + 0] + noise, 0, 255);
    rgba_image[idx + 1] =
        (unsigned char)Clampf(rgba_image[idx + 1] + noise, 0, 255);
    rgba_image[idx + 2] =
        (unsigned char)Clampf(rgba_image[idx + 2] + noise, 0, 255);
  }
}

// =====================================================
// CHANGE 6: LoadImage — split RGB into Y, Cb, Cr channels
// =====================================================
bool LoadImage(const char *filename) {
  int n;
  unsigned char *data = stbi_load(filename, &g_tex_w, &g_tex_h, &n, 4);
  if (!data)
    return false;

  int npx = g_tex_w * g_tex_h;
  g_rgba_orig.assign(data, data + npx * 4);

  // Split into three channel buffers
  g_Y.resize(npx);
  g_Cb.resize(npx);
  g_Cr.resize(npx);
  for (int i = 0; i < npx; i++) {
    unsigned char r = data[i * 4 + 0], gv = data[i * 4 + 1],
                  b = data[i * 4 + 2];
    g_Y[i] = RGBtoY(r, gv, b);
    g_Cb[i] = RGBtoCb(r, gv, b);
    g_Cr[i] = RGBtoCr(r, gv, b);
  }

  // DCT blocks for all three channels
  g_dct_blocks_Y = ComputeDCTBlocks(g_Y, g_tex_w, g_tex_h, g_block_size);
  g_dct_blocks_Cb = ComputeDCTBlocks(g_Cb, g_tex_w, g_tex_h, g_block_size);
  g_dct_blocks_Cr = ComputeDCTBlocks(g_Cr, g_tex_w, g_tex_h, g_block_size);

  g_selected_block = -1;
  g_experiment_run = false;
  g_chain_run = false;
  g_ctx_model.Init(g_block_size);

  DeleteTextureSafe(g_tex_original);
  g_tex_original = UploadTexture(data, g_tex_w, g_tex_h);
  stbi_image_free(data);
  return true;
}

// =====================================================
// CHANGE 7: QuantizeChannel helper (new function)
// =====================================================
static std::vector<QuantResult>
QuantizeChannel(const std::vector<DCTBlock> &blocks,
                const std::vector<unsigned char> &ref, int blockSize,
                QuantMode mode, float baseQ, float deadzoneScale,
                bool useTrellis, float lambda, bool signHiding,
                std::vector<float> &out_recon) {
  out_recon.assign(g_tex_w * g_tex_h, 128.0f);
  std::vector<QuantResult> results;
  for (auto &block : blocks) {
    QuantResult qres = QuantizeBlock(block, blockSize, mode, baseQ,
                                     deadzoneScale, useTrellis, lambda);
    if (signHiding)
      ApplySignHidingToResult(qres);
    qres.psnr = ComputePSNR(ref, qres.reconPixels, block.bx, block.by,
                            blockSize, g_tex_w, g_tex_h);
    for (int r = 0; r < blockSize; r++)
      for (int c = 0; c < blockSize; c++) {
        int ix = block.bx * blockSize + c, iy = block.by * blockSize + r;
        if (ix >= g_tex_w || iy >= g_tex_h)
          continue;
        out_recon[iy * g_tex_w + ix] = qres.reconPixels[r * blockSize + c];
      }
    results.push_back(qres);
  }
  return results;
}

// =====================================================
// CHANGE 8: RunExperiment — quantize Y, Cb, Cr separately
// =====================================================
void RunExperiment() {
  if (g_dct_blocks_Y.empty())
    return;

  g_quant_results.clear();
  g_quant_results_Y.clear();
  g_quant_results_Cb.clear();
  g_quant_results_Cr.clear();
  g_entropy_results.clear();
  g_recon_image.assign(g_tex_w * g_tex_h * 4, 255);
  g_ctx_model.Init(g_block_size);

  float qY = g_base_q;
  float qC = g_base_q + g_chroma_qp_offset;

  g_quant_results_Y = QuantizeChannel(
      g_dct_blocks_Y, g_Y, g_block_size, g_quant_mode, qY, g_deadzone_scale,
      g_use_trellis, g_trellis_lambda, use_sign_hiding, g_recon_Y);
  g_quant_results_Cb = QuantizeChannel(
      g_dct_blocks_Cb, g_Cb, g_block_size, g_quant_mode, qC, g_deadzone_scale,
      g_use_trellis, g_trellis_lambda, use_sign_hiding, g_recon_Cb);
  g_quant_results_Cr = QuantizeChannel(
      g_dct_blocks_Cr, g_Cr, g_block_size, g_quant_mode, qC, g_deadzone_scale,
      g_use_trellis, g_trellis_lambda, use_sign_hiding, g_recon_Cr);

  // Alias Y results so existing tabs (Quantization, Entropy) still work
  g_quant_results = g_quant_results_Y;

  // Entropy on Y channel only
  std::vector<float> all_coeffs;
  for (auto &bl : g_dct_blocks_Y)
    for (float v : bl.coeffs)
      all_coeffs.push_back(v);
  float global_sigma = EstimateLaplacianSigma(all_coeffs);
  for (auto &qres : g_quant_results_Y)
    g_entropy_results.push_back(
        AnalyzeEntropy(qres, g_entropy_model, g_ctx_model, global_sigma));

  // Per-channel PSNR
  g_psnr_Y = ChannelPSNR(g_Y, g_recon_Y);
  g_psnr_Cb = ChannelPSNR(g_Cb, g_recon_Cb);
  g_psnr_Cr = ChannelPSNR(g_Cr, g_recon_Cr);

  // Recombine YCbCr -> RGBA color output
  double mse_rgb = 0;
  for (int i = 0; i < g_tex_w * g_tex_h; i++) {
    float Y = g_recon_Y[i], Cb = g_recon_Cb[i], Cr = g_recon_Cr[i];
    unsigned char R = YCbCrtoR(Y, Cb, Cr), G = YCbCrtoG(Y, Cb, Cr),
                  B = YCbCrtoB(Y, Cb, Cr);
    int idx = i * 4;
    g_recon_image[idx + 0] = R;
    g_recon_image[idx + 1] = G;
    g_recon_image[idx + 2] = B;
    g_recon_image[idx + 3] = 255;
    double dr = g_rgba_orig[idx + 0] - R, dg = g_rgba_orig[idx + 1] - G,
           db = g_rgba_orig[idx + 2] - B;
    mse_rgb += (dr * dr + dg * dg + db * db) / 3.0;
  }
  mse_rgb /= (double)(g_tex_w * g_tex_h);
  g_psnr_rgb = (mse_rgb < 1e-10)
                   ? 99.99f
                   : (float)(10.0 * log10(255.0 * 255.0 / mse_rgb));

  if (use_grain)
    ApplyFilmGrain(g_recon_image, g_tex_w, g_tex_h, grain_strength, grain_seed);

  if (g_tex_recon)
    UpdateTexture(g_tex_recon, g_recon_image.data(), g_tex_w, g_tex_h);
  else
    g_tex_recon = UploadTexture(g_recon_image.data(), g_tex_w, g_tex_h);

  // Loop filter on Y only; rebuild RGBA with filtered Y + Cb/Cr
  if (g_lf_enabled) {
    g_lf_result =
        ApplyLoopFilter(g_recon_Y, g_tex_w, g_tex_h, g_block_size, g_lf_params);
    g_lf_result.psnr_before = ChannelPSNR(g_Y, g_recon_Y);
    g_lf_result.psnr_after =
        LFPSNR(g_Y, g_lf_result.filtered, g_tex_w, g_tex_h);
    g_filtered_image.resize(g_tex_w * g_tex_h * 4);
    for (int i = 0; i < g_tex_w * g_tex_h; i++) {
      float Y = g_lf_result.filtered[i], Cb = g_recon_Cb[i], Cr = g_recon_Cr[i];
      int idx = i * 4;
      g_filtered_image[idx + 0] = YCbCrtoR(Y, Cb, Cr);
      g_filtered_image[idx + 1] = YCbCrtoG(Y, Cb, Cr);
      g_filtered_image[idx + 2] = YCbCrtoB(Y, Cb, Cr);
      g_filtered_image[idx + 3] = 255;
    }
    if (g_tex_filtered)
      UpdateTexture(g_tex_filtered, g_filtered_image.data(), g_tex_w, g_tex_h);
    else
      g_tex_filtered = UploadTexture(g_filtered_image.data(), g_tex_w, g_tex_h);
  }

  g_experiment_run = true;
  g_chain_run = false;
}

// =====================================================
// CHANGE 9: RunChainPipeline — use Y float buffer, recombine color
// =====================================================
void RunChainPipeline() {
  if (!g_experiment_run || g_recon_Y.empty())
    return;

  g_chain_results =
      RunChain(g_recon_Y, g_Y, g_tex_w, g_tex_h, g_block_size, g_chain_steps);

  if (!g_chain_results.empty()) {
    auto &final_stage = g_chain_results.back();
    std::vector<unsigned char> out_rgba(g_tex_w * g_tex_h * 4);
    for (int i = 0; i < g_tex_w * g_tex_h; i++) {
      float Y = final_stage.pixels[i], Cb = g_recon_Cb[i], Cr = g_recon_Cr[i];
      int idx = i * 4;
      out_rgba[idx + 0] = YCbCrtoR(Y, Cb, Cr);
      out_rgba[idx + 1] = YCbCrtoG(Y, Cb, Cr);
      out_rgba[idx + 2] = YCbCrtoB(Y, Cb, Cr);
      out_rgba[idx + 3] = 255;
    }
    if (g_tex_chain_out)
      UpdateTexture(g_tex_chain_out, out_rgba.data(), g_tex_w, g_tex_h);
    else
      g_tex_chain_out = UploadTexture(out_rgba.data(), g_tex_w, g_tex_h);
  }
  g_chain_run = true;
}

// =====================================================
// EXPORT — CSV and PDF
// =====================================================
static std::string Timestamp() {
  time_t t = time(nullptr);
  char buf[32];
  strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&t));
  return std::string(buf);
}

static void ExportQuantCSV() {
  if (!g_experiment_run || g_selected_block < 0 ||
      g_selected_block >= (int)g_quant_results_Y.size())
    return;
  char path[MAX_PATH] = "";
  std::string def = "quant_block_" + Timestamp() + ".csv";
  strncpy(path, def.c_str(), MAX_PATH - 1);
  if (!SaveFileDialog(path, MAX_PATH, "CSV Files\0*.csv\0All Files\0*.*\0",
                      "csv"))
    return;
  QuantResult &res = g_quant_results_Y[g_selected_block];
  DCTBlock &blk = g_dct_blocks_Y[g_selected_block];
  int N = g_block_size;
  std::ofstream f(path);
  if (!f.is_open())
    return;
  f << "# Codec Cooker — Quantization Export (Y channel)\n";
  f << "# Block," << blk.bx << "," << blk.by << "\n";
  f << "# PSNR_Y," << g_psnr_Y << "\n# PSNR_Cb," << g_psnr_Cb << "\n# PSNR_Cr,"
    << g_psnr_Cr << "\n# PSNR_RGB," << g_psnr_rgb << "\n";
  f << "# BaseQ_Y," << g_base_q << "\n# ChromaQPOffset," << g_chroma_qp_offset
    << "\n# BaseQ_Chroma," << (g_base_q + g_chroma_qp_offset) << "\n#\n";
  f << "Idx,Row,Col,Original,Step,Level,Reconstructed,Error\n";
  for (int i = 0; i < (int)res.entries.size(); i++) {
    auto &en = res.entries[i];
    f << i << "," << (i / N) << "," << (i % N) << "," << en.original << ","
      << en.step << "," << en.level << "," << en.reconstructed << ","
      << en.error << "\n";
  }
  f.close();
}

static void ExportEntropyCSV() {
  if (!g_experiment_run || g_selected_block < 0 ||
      g_selected_block >= (int)g_entropy_results.size())
    return;
  char path[MAX_PATH] = "";
  std::string def = "entropy_block_" + Timestamp() + ".csv";
  strncpy(path, def.c_str(), MAX_PATH - 1);
  if (!SaveFileDialog(path, MAX_PATH, "CSV Files\0*.csv\0All Files\0*.*\0",
                      "csv"))
    return;
  EntropyResult &er = g_entropy_results[g_selected_block];
  DCTBlock &blk = g_dct_blocks_Y[g_selected_block];
  int N = g_block_size;
  std::ofstream f(path);
  if (!f.is_open())
    return;
  f << "# Codec Cooker — Entropy Export\n";
  f << "# Block," << blk.bx << "," << blk.by << "\n";
  f << "# NaiveBits," << er.total_bits_naive << "\n";
  f << "# ModelBits," << er.total_bits_model << "\n";
  f << "# RLEBits," << er.total_bits_rle << "\n";
  f << "# Nonzero," << er.nonzero_count << "/" << N * N << "\n";
  f << "# EntropyModel," << (int)g_entropy_model << "\n#\n";
  f << "Idx,Row,Col,Level,Probability,BitCost,LaplaceSigma\n";
  for (auto &en : er.entries)
    f << en.idx << "," << en.row << "," << en.col << "," << en.level << ","
      << en.probability << "," << en.bit_cost << "," << en.laplace_sigma
      << "\n";
  f << "\n# RLE Sequence (zigzag order)\nRun,Level,Type\n";
  for (auto &rle : er.rle) {
    f << rle.run << ",";
    if (rle.level == INT_MIN)
      f << "EOB,end\n";
    else
      f << rle.level << ",coeff\n";
  }
  f.close();
}

// WritePDF, ExportQuantPDF, ExportEntropyPDF, and main() continue in main2.cpp
//
// =====================================================
// Continuation of main.cpp — paste after main1.cpp
// =====================================================

static void WritePDF(const char *path, const std::string &title,
                     const std::vector<std::string> &lines) {
  auto escape = [](const std::string &s) {
    std::string out;
    for (char c : s) {
      if (c == '(' || c == ')' || c == '\\')
        out += '\\';
      out += c;
    }
    return out;
  };
  std::ostringstream content;
  content << "BT\n/F1 9 Tf\n";
  float y = 770.0f, lh = 11.0f;
  content << "50 " << y << " Td\n/F1 11 Tf\n(" << escape(title)
          << ") Tj\n/F1 9 Tf\n";
  y -= lh * 2;
  for (auto &line : lines) {
    if (y < 40.0f) {
      content << "50 " << y << " Td\n(...truncated...) Tj\n";
      break;
    }
    content << "50 " << y << " Td\n(" << escape(line) << ") Tj\n";
    y -= lh;
  }
  content << "ET\n";
  std::string stream = content.str();
  std::ofstream f(path, std::ios::binary);
  if (!f.is_open())
    return;
  std::vector<long> offsets(6, 0);
  auto pos = [&]() -> long { return (long)f.tellp(); };
  f << "%PDF-1.4\n";
  offsets[1] = pos();
  f << "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n";
  offsets[2] = pos();
  f << "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n";
  offsets[3] = pos();
  f << "3 0 obj\n<< /Type /Page /Parent 2 0 R\n"
       "   /MediaBox [0 0 595 842]\n   /Contents 5 0 R\n"
       "   /Resources << /Font << /F1 4 0 R >> >> >>\nendobj\n";
  offsets[4] = pos();
  f << "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier "
       ">>\nendobj\n";
  offsets[5] = pos();
  f << "5 0 obj\n<< /Length " << stream.size() << " >>\nstream\n"
    << stream << "\nendstream\nendobj\n";
  long xp = pos();
  f << "xref\n0 6\n0000000000 65535 f \n";
  for (int i = 1; i <= 5; i++) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%010ld 00000 n \n", offsets[i]);
    f << buf;
  }
  f << "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n" << xp << "\n%%EOF\n";
  f.close();
}

static void ExportQuantPDF() {
  if (!g_experiment_run || g_selected_block < 0 ||
      g_selected_block >= (int)g_quant_results_Y.size())
    return;
  char path[MAX_PATH] = "";
  std::string def = "quant_block_" + Timestamp() + ".pdf";
  strncpy(path, def.c_str(), MAX_PATH - 1);
  if (!SaveFileDialog(path, MAX_PATH, "PDF Files\0*.pdf\0All Files\0*.*\0",
                      "pdf"))
    return;
  QuantResult &res = g_quant_results_Y[g_selected_block];
  DCTBlock &blk = g_dct_blocks_Y[g_selected_block];
  int N = g_block_size;
  std::string title = "Quantization Report Block (" + std::to_string(blk.bx) +
                      "," + std::to_string(blk.by) + ")";
  std::vector<std::string> lines;
  char buf[128];
  snprintf(buf, sizeof(buf),
           "Y PSNR: %.2f dB  Cb PSNR: %.2f dB  Cr PSNR: %.2f dB  RGB: %.2f dB",
           g_psnr_Y, g_psnr_Cb, g_psnr_Cr, g_psnr_rgb);
  lines.push_back(buf);
  snprintf(buf, sizeof(buf),
           "Base Q Y: %.0f  Chroma Offset: +%.0f  Effective Chroma Q: %.0f",
           g_base_q, g_chroma_qp_offset, g_base_q + g_chroma_qp_offset);
  lines.push_back(buf);
  lines.push_back("Sign Hiding: " +
                  (std::string)(use_sign_hiding ? "yes" : "no"));
  lines.push_back("");
  lines.push_back(
      "Idx   [r,c]   Original      Step     Level   Reconstructed   Error");
  lines.push_back(std::string(70, '-'));
  for (int i = 0; i < (int)res.entries.size(); i++) {
    auto &en = res.entries[i];
    snprintf(buf, sizeof(buf),
             "%-5d [%d,%-2d]  %10.3f  %8.2f  %6d  %12.3f  %10.4f", i, i / N,
             i % N, en.original, en.step, en.level, en.reconstructed, en.error);
    lines.push_back(buf);
  }
  WritePDF(path, title, lines);
}

static void ExportEntropyPDF() {
  if (!g_experiment_run || g_selected_block < 0 ||
      g_selected_block >= (int)g_entropy_results.size())
    return;
  char path[MAX_PATH] = "";
  std::string def = "entropy_block_" + Timestamp() + ".pdf";
  strncpy(path, def.c_str(), MAX_PATH - 1);
  if (!SaveFileDialog(path, MAX_PATH, "PDF Files\0*.pdf\0All Files\0*.*\0",
                      "pdf"))
    return;
  EntropyResult &er = g_entropy_results[g_selected_block];
  DCTBlock &blk = g_dct_blocks_Y[g_selected_block];
  int N = g_block_size;
  std::string title = "Entropy Report Block (" + std::to_string(blk.bx) + "," +
                      std::to_string(blk.by) + ")";
  std::vector<std::string> lines;
  char buf[128];
  snprintf(buf, sizeof(buf), "Naive: %.1f  Model: %.1f  RLE: %.1f",
           er.total_bits_naive, er.total_bits_model, er.total_bits_rle);
  lines.push_back(buf);
  snprintf(buf, sizeof(buf), "Nonzero: %d / %d", er.nonzero_count, N * N);
  lines.push_back(buf);
  lines.push_back("");
  lines.push_back("[r,c]   Level   Probability    BitCost   Sigma");
  lines.push_back(std::string(55, '-'));
  for (auto &en : er.entries) {
    snprintf(buf, sizeof(buf), "[%d,%-2d]  %5d   %.6f   %7.3f   %6.2f", en.row,
             en.col, en.level, en.probability, en.bit_cost, en.laplace_sigma);
    lines.push_back(buf);
  }
  lines.push_back("");
  lines.push_back("RLE Sequence:");
  lines.push_back(std::string(28, '-'));
  for (auto &rle : er.rle) {
    if (rle.level == INT_MIN)
      snprintf(buf, sizeof(buf), "%-5d EOB     end", rle.run);
    else
      snprintf(buf, sizeof(buf), "%-5d %-7d coeff", rle.run, rle.level);
    lines.push_back(buf);
  }
  WritePDF(path, title, lines);
}

// =====================================================
// MAIN
// =====================================================
int main(int, char **) {
  SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

  SDL_Window *window = SDL_CreateWindow(
      "Codec Cooker", 1600, 1000,
      SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
  SDL_GLContext gl_context = SDL_GL_CreateContext(window);
  SDL_GL_MakeCurrent(window, gl_context);
  SDL_GL_SetSwapInterval(1);

  // Load window icon relative to exe directory
  {
    char exe_dir[MAX_PATH] = "";
    GetModuleFileNameA(NULL, exe_dir, MAX_PATH);
    char *slash = strrchr(exe_dir, '\\');
    if (!slash)
      slash = strrchr(exe_dir, '/');
    std::string icon_path;
    if (slash) {
      *slash = '\0';
      icon_path = std::string(exe_dir) + "\\codec-cooker-icon.png";
    } else
      icon_path = "codec-cooker-icon.png";
    int iw, ih, ich;
    unsigned char *icon = stbi_load(icon_path.c_str(), &iw, &ih, &ich, 4);
    if (icon) {
      SDL_Surface *surf =
          SDL_CreateSurfaceFrom(iw, ih, SDL_PIXELFORMAT_RGBA32, icon, iw * 4);
      if (surf) {
        SDL_SetWindowIcon(window, surf);
        SDL_DestroySurface(surf);
      }
      g_icon_tex = UploadTexture(icon, iw, ih);
      g_icon_w = iw;
      g_icon_h = ih;
      stbi_image_free(icon);
    }
  }

  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();

  ImFont *font_default = io.Fonts->AddFontDefault();
  ImFont *font_mono = nullptr, *font_large = nullptr;
  const char *mono_paths[] = {"C:/Windows/Fonts/consola.ttf",
                              "C:/Windows/Fonts/cour.ttf", nullptr};
  for (int i = 0; mono_paths[i]; i++) {
    FILE *f = fopen(mono_paths[i], "rb");
    if (f) {
      fclose(f);
      font_mono = io.Fonts->AddFontFromFileTTF(mono_paths[i], 14.0f);
      break;
    }
  }
  const char *sans_paths[] = {"C:/Windows/Fonts/segoeui.ttf",
                              "C:/Windows/Fonts/arial.ttf", nullptr};
  for (int i = 0; sans_paths[i]; i++) {
    FILE *f = fopen(sans_paths[i], "rb");
    if (f) {
      fclose(f);
      font_large = io.Fonts->AddFontFromFileTTF(sans_paths[i], 15.0f);
      break;
    }
  }

  std::vector<const char *> font_names = {"Default"};
  std::vector<ImFont *> fonts = {font_default};
  if (font_mono) {
    font_names.push_back("Monospace");
    fonts.push_back(font_mono);
  }
  if (font_large) {
    font_names.push_back("Sans-Serif");
    fonts.push_back(font_large);
  }

  ImGui::StyleColorsDark();
  ImGuiStyle &style = ImGui::GetStyle();
  ApplyTheme(style);
  g_ctx_model.Init(g_block_size);

  g_chain_steps = {
      {CHAIN_DEBLOCK, true, 1.0f, 16.0f, 8.0f},
      {CHAIN_CDEF, true, 1.0f},
      {CHAIN_WIENER, true, 1.0f, 16.0f, 8.0f, 2.0f, 20.0f, 3},
  };

  ImGui_ImplSDL3_InitForOpenGL(window, gl_context);
  ImGui_ImplOpenGL3_Init("#version 130");

  bool done = false, show_settings = false;
  char file_path[260] = "";

  while (!done) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      ImGui_ImplSDL3_ProcessEvent(&e);
      if (e.type == SDL_EVENT_QUIT)
        done = true;
    }
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    if (g_theme.font_idx < (int)fonts.size() && fonts[g_theme.font_idx])
      ImGui::PushFont(fonts[g_theme.font_idx]);

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::Begin("Codec Cooker", nullptr,
                 ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

    // MENU BAR
    if (ImGui::BeginMenuBar()) {
      if (g_icon_tex) {
        float icon_h = ImGui::GetFrameHeight() - 2;
        float icon_w =
            icon_h * (float)g_icon_w / (float)(g_icon_h ? g_icon_h : 1);
        ImGui::Image((ImTextureID)(intptr_t)g_icon_tex, ImVec2(icon_w, icon_h));
        ImGui::SameLine();
      }
      ImGui::Text("Codec Cooker");
      ImGui::SameLine(0, 16);
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Open Image")) {
          if (OpenFileDialog(file_path, sizeof(file_path))) {
            ClearAll();
            LoadImage(file_path);
          }
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Close"))
          done = true;
        ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("Settings")) {
        if (ImGui::MenuItem("Theme Editor"))
          show_settings = true;
        ImGui::EndMenu();
      }
      ImGui::EndMenuBar();
    }

    // THEME EDITOR
    if (show_settings) {
      ImGui::SetNextWindowSize(ImVec2(460, 520), ImGuiCond_FirstUseEver);
      ImGui::Begin("Theme Editor", &show_settings);
      bool changed = false;
      ImGui::SeparatorText("Colors");
      float p[3] = {g_theme.primary.x, g_theme.primary.y, g_theme.primary.z};
      float b[3] = {g_theme.bg.x, g_theme.bg.y, g_theme.bg.z};
      float t[3] = {g_theme.text.x, g_theme.text.y, g_theme.text.z};
      float sp[3] = {g_theme.separator.x, g_theme.separator.y,
                     g_theme.separator.z};
      float tb[3] = {g_theme.tab_bg.x, g_theme.tab_bg.y, g_theme.tab_bg.z};
      if (ImGui::ColorEdit3("Accent / Primary", p)) {
        g_theme.primary = ImVec4(p[0], p[1], p[2], 1);
        changed = true;
      }
      if (ImGui::ColorEdit3("Background", b)) {
        g_theme.bg = ImVec4(b[0], b[1], b[2], 1);
        changed = true;
      }
      if (ImGui::ColorEdit3("Text", t)) {
        g_theme.text = ImVec4(t[0], t[1], t[2], 1);
        changed = true;
      }
      if (ImGui::ColorEdit3("Section Lines", sp)) {
        g_theme.separator = ImVec4(sp[0], sp[1], sp[2], 1);
        changed = true;
      }
      if (ImGui::ColorEdit3("Tab Background", tb)) {
        g_theme.tab_bg = ImVec4(tb[0], tb[1], tb[2], 1);
        changed = true;
      }
      ImGui::SeparatorText("Typography");
      if (ImGui::Combo("Font", &g_theme.font_idx, font_names.data(),
                       (int)font_names.size()))
        changed = true;
      ImGui::SliderFloat("Font Scale", &g_theme.font_scale, 0.5f, 2.5f);
      io.FontGlobalScale = g_theme.font_scale;
      if (changed)
        ApplyTheme(style);
      ImGui::SeparatorText("Preview");
      ImGui::TextColored(g_theme.primary, "Primary color text");
      ImGui::Text("Normal text sample");
      ImGui::Separator();
      ImGui::Text("Section separator above ^");
      if (ImGui::Button("Sample Button")) {
      }
      ImGui::End();
    }

    // =====================================================
    // LEFT PANEL
    // =====================================================
    ImGui::BeginChild("Left", ImVec2(280, 0), true);

    if (ImGui::Button("Open Image", ImVec2(-1, 32))) {
      if (OpenFileDialog(file_path, sizeof(file_path))) {
        ClearAll();
        LoadImage(file_path);
      }
    }
    if (g_tex_original) {
      if (ImGui::Button("Clear / New Image", ImVec2(-1, 28)))
        ClearAll();
    }

    ImGui::SeparatorText("Block");
    static int bs_idx = 1;
    const char *bs_opts[] = {"4x4", "8x8", "16x16", "32x32"};
    const int bs_vals[] = {4, 8, 16, 32};
    if (ImGui::Combo("Size##bs", &bs_idx, bs_opts, 4)) {
      g_block_size = bs_vals[bs_idx];
      if (!g_Y.empty()) {
        // CHANGE 13: recompute all three channels on block size change
        g_dct_blocks_Y = ComputeDCTBlocks(g_Y, g_tex_w, g_tex_h, g_block_size);
        g_dct_blocks_Cb =
            ComputeDCTBlocks(g_Cb, g_tex_w, g_tex_h, g_block_size);
        g_dct_blocks_Cr =
            ComputeDCTBlocks(g_Cr, g_tex_w, g_tex_h, g_block_size);
        g_ctx_model.Init(g_block_size);
      }
    }

    ImGui::SeparatorText("Quantization");
    const char *qmode_names[] = {"Flat", "JPEG Perceptual", "Ramp", "Deadzone",
                                 "Custom"};
    int qm = (int)g_quant_mode;
    if (ImGui::Combo("Mode##qm", &qm, qmode_names, 5))
      g_quant_mode = (QuantMode)qm;

    // CHANGE 10: Chroma QP offset slider (renamed Base Q to make room)
    ImGui::SliderFloat("Base Q (Y)", &g_base_q, 1.0f, 128.0f);
    ImGui::SliderFloat("Chroma QP Offset", &g_chroma_qp_offset, 0.0f, 32.0f);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip(
          "Added to Base Q for Cb and Cr.\nPositive = coarser chroma "
          "(standard).\nAV1 default ~4-8. Set 0 for equal quality.");
    ImGui::Text("  Cb/Cr Q = %.0f + %.0f = %.0f", g_base_q, g_chroma_qp_offset,
                g_base_q + g_chroma_qp_offset);

    if (g_quant_mode == QUANT_DEADZONE)
      ImGui::SliderFloat("Deadzone", &g_deadzone_scale, 1.0f, 4.0f);
    ImGui::Checkbox("Trellis RDO", &g_use_trellis);
    if (g_use_trellis)
      ImGui::SliderFloat("Lambda", &g_trellis_lambda, 0.01f, 5.0f);

    ImGui::SeparatorText("Entropy");
    const char *emode_names[] = {"Flat Prob", "Laplacian", "Context (AV1)",
                                 "Adaptive"};
    int em = (int)g_entropy_model;
    if (ImGui::Combo("Model##em", &em, emode_names, 4))
      g_entropy_model = (EntropyModel)em;
    if (g_entropy_model == ENTROPY_FIXED_PROB)
      ImGui::SliderFloat("Sigma", &g_fixed_sigma, 1.0f, 200.0f);

    ImGui::SeparatorText("Loop Filter");
    ImGui::Checkbox("Enable##lf", &g_lf_enabled);
    if (g_lf_enabled) {
      const char *lf_names[] = {"None", "Deblock", "Adaptive (CDEF)",
                                "Bilateral", "Wiener"};
      int lm = (int)g_lf_params.mode;
      if (ImGui::Combo("Filter##lf", &lm, lf_names, 5))
        g_lf_params.mode = (LoopFilterMode)lm;
      ImGui::SliderFloat("Strength", &g_lf_params.strength, 0.0f, 4.0f);
      if (g_lf_params.mode == LF_DEBLOCK) {
        ImGui::SliderFloat("Threshold", &g_lf_params.threshold, 1.0f, 64.0f);
        ImGui::SliderFloat("Flat Thresh", &g_lf_params.flat_thresh, 1.0f,
                           32.0f);
      }
      if (g_lf_params.mode == LF_BILATERAL) {
        ImGui::SliderFloat("Sigma Space", &g_lf_params.bilateral_sigma_space,
                           0.5f, 8.0f);
        ImGui::SliderFloat("Sigma Color", &g_lf_params.bilateral_sigma_color,
                           1.0f, 80.0f);
      }
      if (g_lf_params.mode == LF_WIENER)
        ImGui::SliderInt("Radius", &g_lf_params.wiener_radius, 1, 7);
    }

    ImGui::SeparatorText("Other");
    ImGui::Checkbox("Sign Hiding", &use_sign_hiding);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("IMPLEMENTED: Flips last nonzero coefficient sign\n"
                        "to make sign parity even. Saves ~1 bit/block.\n"
                        "H.265/AV1 technique.");
    ImGui::Checkbox("CCSO", &use_ccso);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("STUB: Cross-Component Sample Offset (AV1).\n"
                        "Uses luma values to predict chroma offsets.\n"
                        "Requires YCbCr pipeline — not yet implemented.");
    ImGui::SameLine();
    ImGui::Checkbox("GDF", &use_gdf);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("STUB: Guided Deblocking Filter.\n"
                        "Uses a guidance image (e.g. low-res preview)\n"
                        "to steer the deblock filter. Not yet implemented.");
    ImGui::Checkbox("Film Grain", &use_grain);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("IMPLEMENTED: Adds synthetic Gaussian noise\n"
                        "to reconstructed image (AV1 film grain synthesis).");
    if (use_grain) {
      ImGui::SliderFloat("Grain Strength", &grain_strength, 0.0f, 32.0f);
      ImGui::SliderInt("Grain Seed", &grain_seed, 0, 9999);
    }
    ImGui::Checkbox("256px Superblocks", &use_superblocks);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Changes the Heatmaps tab to show\n"
                        "256x256 superblock groupings instead of\n"
                        "individual coding blocks (visual only).");

    ImGui::Separator();
    bool can_run = !g_dct_blocks_Y.empty();
    if (!can_run)
      ImGui::BeginDisabled();
    if (ImGui::Button("Run Experiment", ImVec2(-1, 40)))
      RunExperiment();
    if (!can_run)
      ImGui::EndDisabled();

    // CHANGE 11: Summary stats — replace Avg PSNR with per-channel PSNR
    if (g_experiment_run) {
      ImGui::Separator();
      ImGui::TextColored(ImVec4(1, 1, 0.4f, 1), "Y  PSNR: %.2f dB", g_psnr_Y);
      ImGui::TextColored(ImVec4(0.4f, 0.8f, 1, 1), "Cb PSNR: %.2f dB",
                         g_psnr_Cb);
      ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "Cr PSNR: %.2f dB",
                         g_psnr_Cr);
      ImGui::Text("RGB PSNR: %.2f dB", g_psnr_rgb);
      if (!g_quant_results_Y.empty()) {
        double tq = 0, te = 0, trle = 0;
        int tnz = 0;
        for (int i = 0; i < (int)g_quant_results_Y.size(); i++) {
          tq += g_quant_results_Y[i].estimatedBits;
          tnz += g_quant_results_Y[i].nonzeroCount;
          if (i < (int)g_entropy_results.size()) {
            te += g_entropy_results[i].total_bits_model;
            trle += g_entropy_results[i].total_bits_rle;
          }
        }
        ImGui::Text("Quant bits:  %.0f", tq);
        ImGui::Text("Model bits:  %.0f", te);
        ImGui::Text("RLE bits:    %.0f", trle);
        ImGui::Text("Nonzero Y:   %d", tnz);
      }
      if (use_sign_hiding)
        ImGui::TextColored(ImVec4(0.4f, 1, 0.4f, 1),
                           "Sign hiding: active (-1 bit/blk)");
      if (g_lf_enabled) {
        ImGui::Separator();
        ImGui::Text("LF before: %.2f dB", g_lf_result.psnr_before);
        ImGui::Text("LF after:  %.2f dB", g_lf_result.psnr_after);
        ImGui::Text("Avg edge:  %.3f", g_lf_result.avg_boundary_strength);
      }
    }

    ImGui::EndChild();
    ImGui::SameLine();

    // =====================================================
    // MAIN PANEL — tabs
    // =====================================================
    ImGui::BeginChild("Main", ImVec2(0, 0), true);
    if (ImGui::BeginTabBar("Tabs")) {

      // TAB: Original
      if (ImGui::BeginTabItem("Original")) {
        if (g_tex_original)
          ImGui::Image((ImTextureID)(intptr_t)g_tex_original,
                       ImVec2((float)g_tex_w, (float)g_tex_h));
        else
          ImGui::TextDisabled("No image loaded.");
        ImGui::EndTabItem();
      }

      // TAB: Blocks
      if (ImGui::BeginTabItem("Blocks")) {
        if (!g_tex_original) {
          ImGui::TextDisabled("Load an image first.");
        } else {
          float panel_w = ImGui::GetContentRegionAvail().x - 290;
          float scale = panel_w / (float)g_tex_w;
          ImVec2 img_pos = ImGui::GetCursorScreenPos();
          ImGui::Image((ImTextureID)(intptr_t)g_tex_original,
                       ImVec2(g_tex_w * scale, g_tex_h * scale));
          ImDrawList *draw = ImGui::GetWindowDrawList();
          float bs = g_block_size * scale;
          float max_energy = 1.0f;
          for (auto &bl : g_dct_blocks_Y)
            if (bl.energy > max_energy)
              max_energy = bl.energy;
          for (int i = 0; i < (int)g_dct_blocks_Y.size(); i++) {
            auto &bl = g_dct_blocks_Y[i];
            float px = img_pos.x + bl.bx * bs, py = img_pos.y + bl.by * bs;
            float norm = bl.energy / max_energy;
            ImU32 fill = (i == g_selected_block) ? IM_COL32(255, 255, 0, 120)
                                                 : HeatColor(norm, 50);
            ImU32 border = (i == g_selected_block)
                               ? IM_COL32(255, 255, 0, 255)
                               : IM_COL32(200, 200, 200, 90);
            draw->AddRectFilled(ImVec2(px, py),
                                ImVec2(px + bs - 1, py + bs - 1), fill);
            draw->AddRect(ImVec2(px, py), ImVec2(px + bs, py + bs), border, 0,
                          0, 1.0f);
          }
          if (use_superblocks) {
            float sb_px = 256.0f * scale;
            for (float sx = img_pos.x; sx < img_pos.x + g_tex_w * scale;
                 sx += sb_px)
              draw->AddLine(ImVec2(sx, img_pos.y),
                            ImVec2(sx, img_pos.y + g_tex_h * scale),
                            IM_COL32(0, 255, 100, 180), 2.0f);
            for (float sy = img_pos.y; sy < img_pos.y + g_tex_h * scale;
                 sy += sb_px)
              draw->AddLine(ImVec2(img_pos.x, sy),
                            ImVec2(img_pos.x + g_tex_w * scale, sy),
                            IM_COL32(0, 255, 100, 180), 2.0f);
          }
          if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0)) {
            ImVec2 mouse = io.MousePos;
            int bx = (int)((mouse.x - img_pos.x) / bs),
                by = (int)((mouse.y - img_pos.y) / bs);
            for (int i = 0; i < (int)g_dct_blocks_Y.size(); i++)
              if (g_dct_blocks_Y[i].bx == bx && g_dct_blocks_Y[i].by == by) {
                g_selected_block = i;
                break;
              }
          }
          ImGui::SameLine();
          ImGui::BeginChild("BlockInspector", ImVec2(280, 0), true);
          ImGui::Text("Block Inspector");
          ImGui::Separator();
          if (g_selected_block >= 0 &&
              g_selected_block < (int)g_dct_blocks_Y.size()) {
            auto &bl = g_dct_blocks_Y[g_selected_block];
            ImGui::Text("Block  (%d, %d)", bl.bx, bl.by);
            ImGui::Text("Energy (Y) %.2f", bl.energy);
            ImGui::Text("Origin (%d, %d)", bl.bx * g_block_size,
                        bl.by * g_block_size);
            if (g_experiment_run &&
                g_selected_block < (int)g_quant_results_Y.size())
              ImGui::Text("PSNR (Y) %.2f dB",
                          g_quant_results_Y[g_selected_block].psnr);
            ImGui::Separator();
            ImGui::Text("Y DCT Coefficients:");
            int N = g_block_size;
            float cell = 252.0f / N;
            ImVec2 gpos = ImGui::GetCursorScreenPos();
            float mx = 0.001f;
            for (auto &v : bl.coeffs)
              if (fabsf(v) > mx)
                mx = fabsf(v);
            for (int r = 0; r < N; r++)
              for (int c = 0; c < N; c++) {
                float val = bl.coeffs[r * N + c];
                ImVec2 p0 = ImVec2(gpos.x + c * cell, gpos.y + r * cell);
                draw->AddRectFilled(p0,
                                    ImVec2(p0.x + cell - 1, p0.y + cell - 1),
                                    HeatColor(fabsf(val) / mx, 220));
              }
            ImGui::Dummy(ImVec2(252, 252));
            if (ImGui::IsMouseHoveringRect(
                    gpos, ImVec2(gpos.x + 252, gpos.y + 252))) {
              int hc = (int)((io.MousePos.x - gpos.x) / cell),
                  hr = (int)((io.MousePos.y - gpos.y) / cell);
              if (hc >= 0 && hc < N && hr >= 0 && hr < N)
                ImGui::SetTooltip("[%d,%d] = %.4f", hr, hc,
                                  bl.coeffs[hr * N + hc]);
            }
          } else
            ImGui::TextDisabled("Click a block to inspect.");
          ImGui::EndChild();
        }
        ImGui::EndTabItem();
      }

      // TAB: Slices
      if (ImGui::BeginTabItem("Slices")) {
        if (!g_tex_original || g_Y.empty()) {
          ImGui::TextDisabled("Load an image first.");
        } else {
          static int slice_row = 0, slice_col = 0;
          static bool show_recon = true, show_filtered = true,
                      show_bounds = true;
          ImGui::SliderInt("Row (Y)", &slice_row, 0, g_tex_h - 1);
          ImGui::SliderInt("Col (X)", &slice_col, 0, g_tex_w - 1);
          ImGui::SameLine(0, 20);
          ImGui::Checkbox("Recon", &show_recon);
          ImGui::SameLine();
          ImGui::Checkbox("Filtered", &show_filtered);
          ImGui::SameLine();
          ImGui::Checkbox("Boundaries", &show_bounds);
          ImVec2 avail = ImGui::GetContentRegionAvail();
          float plot_h = (avail.y - 100) / 2.0f;
          if (plot_h < 80)
            plot_h = 80;
          ImDrawList *dl = ImGui::GetWindowDrawList();
          auto DrawSlice = [&](bool horizontal, int line_idx, float H) {
            ImVec2 p0 = ImGui::GetCursorScreenPos();
            float W = avail.x - 4;
            int span = horizontal ? g_tex_w : g_tex_h;
            dl->AddRectFilled(p0, ImVec2(p0.x + W, p0.y + H),
                              IM_COL32(18, 18, 18, 255));
            auto getGray = [&](int i) -> float {
              int idx = horizontal ? (line_idx * g_tex_w + i)
                                   : (i * g_tex_w + line_idx);
              return (idx >= 0 && idx < (int)g_Y.size()) ? (float)g_Y[idx] : 0;
            };
            auto getRecon = [&](int i) -> float {
              if (g_recon_image.empty())
                return -1;
              int idx = horizontal ? (line_idx * g_tex_w + i)
                                   : (i * g_tex_w + line_idx);
              return (idx >= 0 && idx < (int)g_recon_image.size() / 4)
                         ? (float)g_recon_image[idx * 4]
                         : 0;
            };
            auto getFiltered = [&](int i) -> float {
              if (!g_lf_enabled || g_filtered_image.empty())
                return -1;
              int idx = horizontal ? (line_idx * g_tex_w + i)
                                   : (i * g_tex_w + line_idx);
              return (idx >= 0 && idx < (int)g_filtered_image.size() / 4)
                         ? (float)g_filtered_image[idx * 4]
                         : 0;
            };
            for (int x = 1; x < span; x++) {
              float v0 = getGray(x - 1), v1 = getGray(x);
              float px0 = p0.x + (x - 1) * W / span, px1 = p0.x + x * W / span;
              float py0 = p0.y + H - v0 * H / 255.f,
                    py1 = p0.y + H - v1 * H / 255.f;
              dl->AddLine(ImVec2(px0, py0), ImVec2(px1, py1),
                          IM_COL32(220, 220, 220, 255), 1.2f);
            }
            if (show_recon && g_experiment_run) {
              for (int x = 1; x < span; x++) {
                float v0 = getRecon(x - 1), v1 = getRecon(x);
                if (v0 < 0 || v1 < 0)
                  break;
                float px0 = p0.x + (x - 1) * W / span,
                      px1 = p0.x + x * W / span;
                float py0 = p0.y + H - v0 * H / 255.f,
                      py1 = p0.y + H - v1 * H / 255.f;
                dl->AddLine(ImVec2(px0, py0), ImVec2(px1, py1),
                            IM_COL32(60, 210, 60, 200), 1.2f);
              }
            }
            if (show_filtered && g_lf_enabled && g_experiment_run) {
              for (int x = 1; x < span; x++) {
                float v0 = getFiltered(x - 1), v1 = getFiltered(x);
                if (v0 < 0 || v1 < 0)
                  break;
                float px0 = p0.x + (x - 1) * W / span,
                      px1 = p0.x + x * W / span;
                float py0 = p0.y + H - v0 * H / 255.f,
                      py1 = p0.y + H - v1 * H / 255.f;
                dl->AddLine(ImVec2(px0, py0), ImVec2(px1, py1),
                            IM_COL32(255, 140, 30, 200), 1.2f);
              }
            }
            if (show_bounds) {
              for (int bnd = 0; bnd < span; bnd += g_block_size) {
                float lx = p0.x + bnd * W / span;
                dl->AddLine(ImVec2(lx, p0.y), ImVec2(lx, p0.y + H),
                            IM_COL32(80, 80, 200, 100), 1.0f);
              }
            }
            ImGui::Dummy(ImVec2(W, H));
          };
          ImGui::Separator();
          ImGui::Text("Horizontal slice — row %d", slice_row);
          DrawSlice(true, slice_row, plot_h);
          ImGui::Text("  White=original  Green=reconstructed  Orange=loop "
                      "filtered  Blue=block boundary");
          ImGui::Separator();
          ImGui::Text("Vertical slice — col %d", slice_col);
          DrawSlice(false, slice_col, plot_h);
        }
        ImGui::EndTabItem();
      }

      // TAB: Quantization
      if (ImGui::BeginTabItem("Quantization")) {
        if (!g_experiment_run)
          ImGui::TextDisabled("Run an experiment first.");
        else if (g_selected_block < 0 ||
                 g_selected_block >= (int)g_quant_results_Y.size())
          ImGui::TextDisabled("Click a block in the Blocks tab.");
        else {
          QuantResult &res = g_quant_results_Y[g_selected_block];
          DCTBlock &blk = g_dct_blocks_Y[g_selected_block];
          int N = g_block_size;
          ImGui::Text(
              "Block (%d,%d)  Y PSNR: %.2f dB  Nonzero: %d/%d  Est bits: %.1f",
              blk.bx, blk.by, res.psnr, res.nonzeroCount, N * N,
              res.estimatedBits);
          ImGui::SameLine(ImGui::GetContentRegionAvail().x - 200);
          if (ImGui::Button("Export CSV##q"))
            ExportQuantCSV();
          ImGui::SameLine();
          if (ImGui::Button("Export PDF##q"))
            ExportQuantPDF();
          ImGui::Separator();
          ImVec2 pp = ImGui::GetCursorScreenPos();
          float cell = 12.0f, psz = N * cell;
          ImDrawList *dl = ImGui::GetWindowDrawList();
          for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) {
              int ix = blk.bx * N + c, iy = blk.by * N + r;
              unsigned char pv =
                  (ix < g_tex_w && iy < g_tex_h) ? g_Y[iy * g_tex_w + ix] : 0;
              ImVec2 p0 = ImVec2(pp.x + c * cell, pp.y + r * cell);
              dl->AddRectFilled(p0, ImVec2(p0.x + cell - 1, p0.y + cell - 1),
                                IM_COL32(pv, pv, pv, 255));
            }
          float rx = psz + 16;
          for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) {
              unsigned char pv =
                  (unsigned char)Clampf(res.reconPixels[r * N + c], 0, 255);
              ImVec2 p0 = ImVec2(pp.x + rx + c * cell, pp.y + r * cell);
              dl->AddRectFilled(p0, ImVec2(p0.x + cell - 1, p0.y + cell - 1),
                                IM_COL32(pv, pv, pv, 255));
            }
          float ex = rx + psz + 16;
          for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) {
              int ix = blk.bx * N + c, iy = blk.by * N + r;
              float op = (ix < g_tex_w && iy < g_tex_h)
                             ? (float)g_Y[iy * g_tex_w + ix]
                             : 0;
              float err =
                  fabsf(op - Clampf(res.reconPixels[r * N + c], 0, 255));
              ImVec2 p0 = ImVec2(pp.x + ex + c * cell, pp.y + r * cell);
              dl->AddRectFilled(p0, ImVec2(p0.x + cell - 1, p0.y + cell - 1),
                                HeatColor(err / 64.f, 255));
            }
          ImGui::Dummy(ImVec2(1, psz + 6));
          ImGui::Text("Y Original");
          ImGui::SameLine(psz + 16);
          ImGui::Text("Y Reconstructed");
          ImGui::SameLine(rx + psz + 16);
          ImGui::Text("Error x4 (heat)");
          ImGui::Separator();
          if (ImGui::BeginTable(
                  "CoeffTable", 6,
                  ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                      ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit,
                  ImVec2(0, 300))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Idx", ImGuiTableColumnFlags_WidthFixed,
                                    40);
            ImGui::TableSetupColumn("[r,c]", ImGuiTableColumnFlags_WidthFixed,
                                    55);
            ImGui::TableSetupColumn("Original",
                                    ImGuiTableColumnFlags_WidthFixed, 90);
            ImGui::TableSetupColumn("Step", ImGuiTableColumnFlags_WidthFixed,
                                    65);
            ImGui::TableSetupColumn("Level", ImGuiTableColumnFlags_WidthFixed,
                                    55);
            ImGui::TableSetupColumn("Reconstructed",
                                    ImGuiTableColumnFlags_WidthFixed, 105);
            ImGui::TableHeadersRow();
            for (int i = 0; i < (int)res.entries.size(); i++) {
              auto &en = res.entries[i];
              ImGui::TableNextRow();
              if (en.level == 0)
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                       IM_COL32(35, 35, 35, 255));
              ImGui::TableSetColumnIndex(0);
              ImGui::Text("%d", i);
              ImGui::TableSetColumnIndex(1);
              ImGui::Text("[%d,%d]", i / N, i % N);
              ImGui::TableSetColumnIndex(2);
              if (fabsf(en.original) > 100)
                ImGui::TextColored(ImVec4(1, 0.6f, 0.2f, 1), "%.2f",
                                   en.original);
              else
                ImGui::Text("%.2f", en.original);
              ImGui::TableSetColumnIndex(3);
              ImGui::Text("%.1f", en.step);
              ImGui::TableSetColumnIndex(4);
              if (en.level == 0)
                ImGui::TextDisabled("0");
              else
                ImGui::TextColored(ImVec4(0.4f, 1, 0.4f, 1), "%d", en.level);
              ImGui::TableSetColumnIndex(5);
              ImGui::Text("%.2f", en.reconstructed);
            }
            ImGui::EndTable();
          }
        }
        ImGui::EndTabItem();
      }

      // CHANGE 12: NEW TAB — YCbCr per-channel breakdown
      if (ImGui::BeginTabItem("YCbCr")) {
        if (!g_experiment_run) {
          ImGui::TextDisabled("Run an experiment to see channel breakdown.");
        } else {
          ImGui::SeparatorText("Channel Quality");
          float max_psnr =
              std::max({g_psnr_Y, g_psnr_Cb, g_psnr_Cr, g_psnr_rgb, 50.0f});
          auto PsnrBar = [&](const char *label, float psnr, ImVec4 col) {
            ImGui::TextColored(col, "%-10s %.2f dB", label, psnr);
            ImGui::SameLine(170);
            ImVec2 p = ImGui::GetCursorScreenPos();
            float w = (psnr / max_psnr) * 280.0f;
            ImGui::GetWindowDrawList()->AddRectFilled(
                p, ImVec2(p.x + w, p.y + 14),
                ImGui::ColorConvertFloat4ToU32(
                    ImVec4(col.x, col.y, col.z, 0.7f)));
            ImGui::Dummy(ImVec2(280, 14));
          };
          PsnrBar("Y  (luma)", g_psnr_Y, ImVec4(1, 1, 0.4f, 1));
          PsnrBar("Cb (blue)", g_psnr_Cb, ImVec4(0.4f, 0.8f, 1, 1));
          PsnrBar("Cr (red)", g_psnr_Cr, ImVec4(1, 0.5f, 0.5f, 1));
          PsnrBar("RGB avg", g_psnr_rgb, ImVec4(0.8f, 0.8f, 0.8f, 1));

          ImGui::SeparatorText("QP Parameters per Channel");
          if (ImGui::BeginTable("QpTable", 5,
                                ImGuiTableFlags_Borders |
                                    ImGuiTableFlags_RowBg |
                                    ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("Channel", ImGuiTableColumnFlags_WidthFixed,
                                    90);
            ImGui::TableSetupColumn("Base Q", ImGuiTableColumnFlags_WidthFixed,
                                    70);
            ImGui::TableSetupColumn("QP Offset",
                                    ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn("Effective Q",
                                    ImGuiTableColumnFlags_WidthFixed, 90);
            ImGui::TableSetupColumn("PSNR (dB)",
                                    ImGuiTableColumnFlags_WidthFixed, 90);
            ImGui::TableHeadersRow();
            struct Row {
              const char *ch;
              float bq, off, psnr;
              ImVec4 col;
            };
            Row rows[] = {
                {"Y  (luma)", g_base_q, 0.0f, g_psnr_Y, ImVec4(1, 1, 0.4f, 1)},
                {"Cb (chroma)", g_base_q, g_chroma_qp_offset, g_psnr_Cb,
                 ImVec4(0.4f, 0.8f, 1, 1)},
                {"Cr (chroma)", g_base_q, g_chroma_qp_offset, g_psnr_Cr,
                 ImVec4(1, 0.5f, 0.5f, 1)},
            };
            for (auto &row : rows) {
              ImGui::TableNextRow();
              ImGui::TableSetColumnIndex(0);
              ImGui::TextColored(row.col, "%s", row.ch);
              ImGui::TableSetColumnIndex(1);
              ImGui::Text("%.0f", row.bq);
              ImGui::TableSetColumnIndex(2);
              if (row.off > 0)
                ImGui::TextColored(ImVec4(1, 0.6f, 0.3f, 1), "+%.0f", row.off);
              else
                ImGui::TextDisabled("0");
              ImGui::TableSetColumnIndex(3);
              ImGui::TextColored(row.col, "%.0f", row.bq + row.off);
              ImGui::TableSetColumnIndex(4);
              ImGui::TextColored(row.col, "%.2f", row.psnr);
            }
            ImGui::EndTable();
          }

          ImGui::SeparatorText("Nonzero Coefficients per Channel");
          int nz_Y = 0, nz_Cb = 0, nz_Cr = 0, tot_Y = 0, tot_Cb = 0, tot_Cr = 0;
          for (auto &r : g_quant_results_Y) {
            nz_Y += r.nonzeroCount;
            tot_Y += (int)r.entries.size();
          }
          for (auto &r : g_quant_results_Cb) {
            nz_Cb += r.nonzeroCount;
            tot_Cb += (int)r.entries.size();
          }
          for (auto &r : g_quant_results_Cr) {
            nz_Cr += r.nonzeroCount;
            tot_Cr += (int)r.entries.size();
          }
          if (ImGui::BeginTable("NzTable", 4,
                                ImGuiTableFlags_Borders |
                                    ImGuiTableFlags_RowBg |
                                    ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("Channel", ImGuiTableColumnFlags_WidthFixed,
                                    90);
            ImGui::TableSetupColumn("Nonzero", ImGuiTableColumnFlags_WidthFixed,
                                    80);
            ImGui::TableSetupColumn("Total", ImGuiTableColumnFlags_WidthFixed,
                                    80);
            ImGui::TableSetupColumn("Sparsity",
                                    ImGuiTableColumnFlags_WidthFixed, 90);
            ImGui::TableHeadersRow();
            auto NzRow = [&](const char *ch, int nz, int tot, ImVec4 col) {
              ImGui::TableNextRow();
              float sp = tot > 0 ? (1.0f - (float)nz / tot) * 100.f : 0;
              ImGui::TableSetColumnIndex(0);
              ImGui::TextColored(col, "%s", ch);
              ImGui::TableSetColumnIndex(1);
              ImGui::Text("%d", nz);
              ImGui::TableSetColumnIndex(2);
              ImGui::Text("%d", tot);
              ImGui::TableSetColumnIndex(3);
              ImGui::TextColored(sp > 80 ? ImVec4(0.4f, 1, 0.4f, 1)
                                         : ImVec4(1, 0.8f, 0.3f, 1),
                                 "%.1f%%", sp);
            };
            NzRow("Y  (luma)", nz_Y, tot_Y, ImVec4(1, 1, 0.4f, 1));
            NzRow("Cb", nz_Cb, tot_Cb, ImVec4(0.4f, 0.8f, 1, 1));
            NzRow("Cr", nz_Cr, tot_Cr, ImVec4(1, 0.5f, 0.5f, 1));
            ImGui::EndTable();
          }

          ImGui::SeparatorText("Estimated Bits per Channel");
          double bits_Y = 0, bits_Cb = 0, bits_Cr = 0;
          for (auto &r : g_quant_results_Y)
            bits_Y += r.estimatedBits;
          for (auto &r : g_quant_results_Cb)
            bits_Cb += r.estimatedBits;
          for (auto &r : g_quant_results_Cr)
            bits_Cr += r.estimatedBits;
          double bits_total = bits_Y + bits_Cb + bits_Cr;
          if (ImGui::BeginTable("BitsTable", 3,
                                ImGuiTableFlags_Borders |
                                    ImGuiTableFlags_RowBg |
                                    ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("Channel", ImGuiTableColumnFlags_WidthFixed,
                                    90);
            ImGui::TableSetupColumn("Est Bits",
                                    ImGuiTableColumnFlags_WidthFixed, 90);
            ImGui::TableSetupColumn("% Total", ImGuiTableColumnFlags_WidthFixed,
                                    80);
            ImGui::TableHeadersRow();
            auto BitsRow = [&](const char *ch, double bits, ImVec4 col) {
              ImGui::TableNextRow();
              float pct =
                  bits_total > 0 ? (float)(bits / bits_total) * 100.f : 0;
              ImGui::TableSetColumnIndex(0);
              ImGui::TextColored(col, "%s", ch);
              ImGui::TableSetColumnIndex(1);
              ImGui::Text("%.0f", bits);
              ImGui::TableSetColumnIndex(2);
              ImGui::Text("%.1f%%", pct);
            };
            BitsRow("Y  (luma)", bits_Y, ImVec4(1, 1, 0.4f, 1));
            BitsRow("Cb", bits_Cb, ImVec4(0.4f, 0.8f, 1, 1));
            BitsRow("Cr", bits_Cr, ImVec4(1, 0.5f, 0.5f, 1));
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextDisabled("Total");
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.0f", bits_total);
            ImGui::TableSetColumnIndex(2);
            ImGui::TextDisabled("100%%");
            ImGui::EndTable();
          }
        }
        ImGui::EndTabItem();
      }

      // TAB: Entropy (unchanged except g_dct_blocks alias covers it)
      if (ImGui::BeginTabItem("Entropy")) {
        if (!g_experiment_run)
          ImGui::TextDisabled("Run an experiment first.");
        else if (g_selected_block < 0 ||
                 g_selected_block >= (int)g_entropy_results.size())
          ImGui::TextDisabled("Click a block in the Blocks tab.");
        else {
          EntropyResult &er = g_entropy_results[g_selected_block];
          DCTBlock &bl = g_dct_blocks_Y[g_selected_block];
          int N = g_block_size;
          ImGui::Text("Block (%d,%d)  Nonzero: %d/%d", bl.bx, bl.by,
                      er.nonzero_count, N * N);
          ImGui::Text("Naive: %.1f  Model: %.1f  RLE: %.1f bits",
                      er.total_bits_naive, er.total_bits_model,
                      er.total_bits_rle);
          float saving =
              er.total_bits_naive > 0
                  ? (float)(1.0 - er.total_bits_model / er.total_bits_naive) *
                        100.0f
                  : 0;
          ImGui::TextColored(ImVec4(0.4f, 1, 0.4f, 1),
                             "Model saves %.1f%% vs naive", saving);
          ImGui::SameLine(ImGui::GetContentRegionAvail().x - 200);
          if (ImGui::Button("Export CSV##e"))
            ExportEntropyCSV();
          ImGui::SameLine();
          if (ImGui::Button("Export PDF##e"))
            ExportEntropyPDF();
          ImGui::Separator();
          ImGui::Text("Bit cost per coefficient (hotter = more bits):");
          ImDrawList *dl = ImGui::GetWindowDrawList();
          float cell = 252.0f / N;
          ImVec2 gpos = ImGui::GetCursorScreenPos();
          float max_bc = 0.001f;
          for (auto &en : er.entries)
            if (en.bit_cost > max_bc)
              max_bc = en.bit_cost;
          for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) {
              float bc = er.entries[r * N + c].bit_cost;
              ImVec2 p0 = ImVec2(gpos.x + c * cell, gpos.y + r * cell);
              dl->AddRectFilled(p0, ImVec2(p0.x + cell - 1, p0.y + cell - 1),
                                HeatColor(bc / max_bc, 230));
              if (er.entries[r * N + c].level != 0)
                dl->AddRect(p0, ImVec2(p0.x + cell - 1, p0.y + cell - 1),
                            IM_COL32(255, 255, 255, 180), 0, 0, 1.0f);
            }
          if (ImGui::IsMouseHoveringRect(gpos,
                                         ImVec2(gpos.x + 252, gpos.y + 252))) {
            int hc = (int)((io.MousePos.x - gpos.x) / cell),
                hr = (int)((io.MousePos.y - gpos.y) / cell);
            if (hc >= 0 && hc < N && hr >= 0 && hr < N) {
              auto &en = er.entries[hr * N + hc];
              ImGui::SetTooltip("[%d,%d]  level=%d  prob=%.5f  bits=%.2f", hr,
                                hc, en.level, en.probability, en.bit_cost);
            }
          }
          ImGui::Dummy(ImVec2(252, 252));
          ImGui::Text("White border = nonzero coefficient");
          ImGui::Separator();
          ImGui::Text("Run-Length Sequence (zigzag order):");
          if (ImGui::BeginTable(
                  "RLETable", 3,
                  ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                      ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit,
                  ImVec2(0, 160))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Run", ImGuiTableColumnFlags_WidthFixed,
                                    60);
            ImGui::TableSetupColumn("Level", ImGuiTableColumnFlags_WidthFixed,
                                    70);
            ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed,
                                    70);
            ImGui::TableHeadersRow();
            for (auto &rle : er.rle) {
              ImGui::TableNextRow();
              ImGui::TableSetColumnIndex(0);
              ImGui::Text("%d", rle.run);
              ImGui::TableSetColumnIndex(1);
              if (rle.level == INT_MIN)
                ImGui::TextColored(ImVec4(1, 0.5f, 0.2f, 1), "EOB");
              else
                ImGui::TextColored(ImVec4(0.4f, 1, 0.4f, 1), "%d", rle.level);
              ImGui::TableSetColumnIndex(2);
              ImGui::TextDisabled(rle.level == INT_MIN ? "end" : "coeff");
            }
            ImGui::EndTable();
          }
          ImGui::Separator();
          ImGui::Text("Per-coefficient breakdown:");
          if (ImGui::BeginTable(
                  "EntropyTable", 5,
                  ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                      ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit,
                  ImVec2(0, 200))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("[r,c]", ImGuiTableColumnFlags_WidthFixed,
                                    55);
            ImGui::TableSetupColumn("Level", ImGuiTableColumnFlags_WidthFixed,
                                    55);
            ImGui::TableSetupColumn("Prob", ImGuiTableColumnFlags_WidthFixed,
                                    80);
            ImGui::TableSetupColumn("Bits", ImGuiTableColumnFlags_WidthFixed,
                                    60);
            ImGui::TableSetupColumn("Sigma", ImGuiTableColumnFlags_WidthFixed,
                                    65);
            ImGui::TableHeadersRow();
            for (auto &en : er.entries) {
              ImGui::TableNextRow();
              if (en.level == 0)
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                       IM_COL32(28, 28, 28, 255));
              ImGui::TableSetColumnIndex(0);
              ImGui::Text("[%d,%d]", en.row, en.col);
              ImGui::TableSetColumnIndex(1);
              if (en.level == 0)
                ImGui::TextDisabled("0");
              else
                ImGui::TextColored(ImVec4(0.4f, 1, 0.4f, 1), "%d", en.level);
              ImGui::TableSetColumnIndex(2);
              ImGui::Text("%.5f", en.probability);
              ImGui::TableSetColumnIndex(3);
              ImGui::TextColored(ImVec4(1, 0.6f, 0.3f, 1), "%.2f", en.bit_cost);
              ImGui::TableSetColumnIndex(4);
              ImGui::Text("%.1f", en.laplace_sigma);
            }
            ImGui::EndTable();
          }
        }
        ImGui::EndTabItem();
      }

      // TAB: Loop Filter
      if (ImGui::BeginTabItem("Loop Filter")) {
        if (!g_experiment_run || !g_lf_enabled)
          ImGui::TextDisabled(
              "Enable Loop Filter in left panel, then Run Experiment.");
        else {
          ImGui::Text("PSNR before: %.2f dB", g_lf_result.psnr_before);
          ImGui::SameLine(0, 20);
          float gain = g_lf_result.psnr_after - g_lf_result.psnr_before;
          ImGui::TextColored(
              gain > 0 ? ImVec4(0.4f, 1, 0.4f, 1) : ImVec4(1, 0.4f, 0.4f, 1),
              "After: %.2f dB (%+.2f dB)", g_lf_result.psnr_after, gain);
          ImGui::Text("Avg boundary strength: %.4f",
                      g_lf_result.avg_boundary_strength);
          ImGui::TextDisabled("Deblock uses AV1-style per-edge strength "
                              "modulation. Applied to Y only.");
          ImGui::Separator();
          float half = (ImGui::GetContentRegionAvail().x - 8) / 2.0f;
          float sc = half / (float)g_tex_w, dw = g_tex_w * sc,
                dh = g_tex_h * sc;
          ImGui::Text("Reconstructed (pre-filter)");
          ImGui::SameLine(half + 8);
          ImGui::Text("Loop Filtered");
          if (g_tex_recon)
            ImGui::Image((ImTextureID)(intptr_t)g_tex_recon, ImVec2(dw, dh));
          ImGui::SameLine();
          if (g_tex_filtered)
            ImGui::Image((ImTextureID)(intptr_t)g_tex_filtered, ImVec2(dw, dh));
          ImGui::Separator();
          ImGui::Text("Boundary strength map:");
          ImVec2 mp = ImGui::GetCursorScreenPos();
          float ms = half / g_tex_w, bsz = g_block_size * ms;
          ImDrawList *dl = ImGui::GetWindowDrawList();
          dl->AddRectFilled(mp,
                            ImVec2(mp.x + g_tex_w * ms, mp.y + g_tex_h * ms),
                            IM_COL32(30, 30, 30, 255));
          for (auto &bd : g_lf_result.boundaries) {
            if (bd.horizontal) {
              float x0 = mp.x + bd.bx * bsz, y0 = mp.y + (bd.by + 1) * bsz;
              dl->AddLine(ImVec2(x0, y0), ImVec2(x0 + bsz, y0),
                          HeatColor(bd.strength, 255), 2.0f);
            } else {
              float x0 = mp.x + (bd.bx + 1) * bsz, y0 = mp.y + bd.by * bsz;
              dl->AddLine(ImVec2(x0, y0), ImVec2(x0, y0 + bsz),
                          HeatColor(bd.strength, 255), 2.0f);
            }
          }
          ImGui::Dummy(ImVec2(g_tex_w * ms, g_tex_h * ms));
        }
        ImGui::EndTabItem();
      }

      // TAB: Chain
      if (ImGui::BeginTabItem("Chain")) {
        if (!g_experiment_run) {
          ImGui::TextDisabled(
              "Run an experiment first to get a reconstructed image to chain.");
        } else {
          ImGui::Text(
              "Filter Pipeline  (drag to reorder — add/remove steps below)");
          ImGui::Separator();
          const char *step_names[] = {"Deblock", "CDEF", "Bilateral", "Wiener"};
          for (int i = 0; i < (int)g_chain_steps.size(); i++) {
            auto &step = g_chain_steps[i];
            ImGui::PushID(i);
            ImGui::Checkbox("##en", &step.enabled);
            ImGui::SameLine();
            ImGui::TextColored(step.enabled ? g_theme.primary
                                            : ImVec4(0.5f, 0.5f, 0.5f, 1),
                               "%d: %s", i + 1, step.Label());
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 90);
            if (i > 0 && ImGui::ArrowButton("##up", ImGuiDir_Up))
              std::swap(g_chain_steps[i], g_chain_steps[i - 1]);
            ImGui::SameLine();
            if (i < (int)g_chain_steps.size() - 1 &&
                ImGui::ArrowButton("##dn", ImGuiDir_Down))
              std::swap(g_chain_steps[i], g_chain_steps[i + 1]);
            ImGui::SameLine();
            if (ImGui::Button("X##rm")) {
              g_chain_steps.erase(g_chain_steps.begin() + i);
              ImGui::PopID();
              break;
            }
            if (step.enabled) {
              ImGui::Indent(16);
              ImGui::SliderFloat("Strength##s", &step.strength, 0.0f, 4.0f);
              if (step.type == CHAIN_DEBLOCK) {
                ImGui::SliderFloat("Threshold##t", &step.threshold, 1.0f,
                                   64.0f);
                ImGui::SliderFloat("Flat##f", &step.flat_thresh, 1.0f, 32.0f);
              }
              if (step.type == CHAIN_BILATERAL) {
                ImGui::SliderFloat("Sigma Space##ss",
                                   &step.bilateral_sigma_space, 0.5f, 8.0f);
                ImGui::SliderFloat("Sigma Color##sc",
                                   &step.bilateral_sigma_color, 1.0f, 80.0f);
              }
              if (step.type == CHAIN_WIENER)
                ImGui::SliderInt("Radius##wr", &step.wiener_radius, 1, 7);
              ImGui::Unindent(16);
            }
            ImGui::PopID();
          }
          ImGui::Separator();
          ImGui::Text("Add step:");
          ImGui::SameLine();
          for (int t = 0; t < 4; t++) {
            if (ImGui::Button(step_names[t])) {
              ChainStep ns;
              ns.type = (ChainStepType)t;
              g_chain_steps.push_back(ns);
            }
            if (t < 3)
              ImGui::SameLine();
          }
          ImGui::Separator();
          if (ImGui::Button("Run Chain", ImVec2(-1, 36)))
            RunChainPipeline();
          if (g_chain_run && !g_chain_results.empty()) {
            ImGui::Separator();
            ImGui::Text("Stage Results:");
            if (ImGui::BeginTable("ChainTable", 3,
                                  ImGuiTableFlags_Borders |
                                      ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_SizingFixedFit)) {
              ImGui::TableSetupColumn("Stage", ImGuiTableColumnFlags_WidthFixed,
                                      150);
              ImGui::TableSetupColumn("PSNR (dB)",
                                      ImGuiTableColumnFlags_WidthFixed, 90);
              ImGui::TableSetupColumn("Gain (dB)",
                                      ImGuiTableColumnFlags_WidthFixed, 90);
              ImGui::TableHeadersRow();
              for (auto &stage : g_chain_results) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%s", stage.label.c_str());
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%.2f", stage.psnr);
                ImGui::TableSetColumnIndex(2);
                if (stage.psnr_gain > 0.001f)
                  ImGui::TextColored(ImVec4(0.4f, 1, 0.4f, 1), "+%.3f",
                                     stage.psnr_gain);
                else if (stage.psnr_gain < -0.001f)
                  ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "%.3f",
                                     stage.psnr_gain);
                else
                  ImGui::TextDisabled("0.000");
              }
              ImGui::EndTable();
            }
            ImGui::Separator();
            float half = (ImGui::GetContentRegionAvail().x - 8) / 2.0f;
            float sc = half / (float)g_tex_w, dw = g_tex_w * sc,
                  dh = g_tex_h * sc;
            ImGui::Text("Reconstructed (chain input)");
            ImGui::SameLine(half + 8);
            ImGui::Text("Chain Output (%d stages)",
                        (int)g_chain_results.size() - 1);
            if (g_tex_recon)
              ImGui::Image((ImTextureID)(intptr_t)g_tex_recon, ImVec2(dw, dh));
            ImGui::SameLine();
            if (g_tex_chain_out)
              ImGui::Image((ImTextureID)(intptr_t)g_tex_chain_out,
                           ImVec2(dw, dh));
          }
        }
        ImGui::EndTabItem();
      }

      // TAB: Reconstructed
      if (ImGui::BeginTabItem("Reconstructed")) {
        if (!g_experiment_run)
          ImGui::TextDisabled("Run an experiment first.");
        else {
          float half = (ImGui::GetContentRegionAvail().x - 12) / 2.0f;
          float sc = half / (float)g_tex_w, dw = g_tex_w * sc,
                dh = g_tex_h * sc;
          const char *mnames[] = {"Flat", "JPEG", "Ramp", "Deadzone", "Custom"};
          ImGui::Text("Original (color)");
          ImGui::SameLine(half + 12);
          ImGui::Text("Reconstructed  Q_Y=%.0f  Q_C=%.0f  %s", g_base_q,
                      g_base_q + g_chroma_qp_offset, mnames[(int)g_quant_mode]);
          ImGui::Image((ImTextureID)(intptr_t)g_tex_original, ImVec2(dw, dh));
          ImGui::SameLine();
          if (g_tex_recon)
            ImGui::Image((ImTextureID)(intptr_t)g_tex_recon, ImVec2(dw, dh));
        }
        ImGui::EndTabItem();
      }

      // TAB: Heatmaps
      if (ImGui::BeginTabItem("Heatmaps")) {
        if (g_dct_blocks_Y.empty())
          ImGui::TextDisabled("Load an image first.");
        else {
          ImGui::Text("DCT Energy per block (Y channel)");
          if (use_superblocks) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0, 1, 0.5f, 1),
                               "  [256px Superblock boundaries shown]");
          }
          float pw = ImGui::GetContentRegionAvail().x;
          float sc = pw / (float)g_tex_w, bs = g_block_size * sc;
          ImVec2 base = ImGui::GetCursorScreenPos();
          ImGui::Dummy(ImVec2(g_tex_w * sc, g_tex_h * sc));
          ImDrawList *dl = ImGui::GetWindowDrawList();
          float mx = 1.0f;
          for (auto &bl : g_dct_blocks_Y)
            if (bl.energy > mx)
              mx = bl.energy;
          for (auto &bl : g_dct_blocks_Y) {
            float px = base.x + bl.bx * bs, py = base.y + bl.by * bs;
            dl->AddRectFilled(ImVec2(px, py), ImVec2(px + bs, py + bs),
                              HeatColor(bl.energy / mx, 230));
            dl->AddRect(ImVec2(px, py), ImVec2(px + bs, py + bs),
                        IM_COL32(0, 0, 0, 60));
          }
          if (use_superblocks) {
            float sb_px = 256.0f * sc;
            for (float sx = base.x; sx < base.x + g_tex_w * sc; sx += sb_px)
              dl->AddLine(ImVec2(sx, base.y), ImVec2(sx, base.y + g_tex_h * sc),
                          IM_COL32(0, 255, 120, 200), 2.0f);
            for (float sy = base.y; sy < base.y + g_tex_h * sc; sy += sb_px)
              dl->AddLine(ImVec2(base.x, sy), ImVec2(base.x + g_tex_w * sc, sy),
                          IM_COL32(0, 255, 120, 200), 2.0f);
          }
        }
        ImGui::EndTabItem();
      }

      ImGui::EndTabBar();
    }

    ImGui::EndChild();
    ImGui::End();

    if (g_theme.font_idx < (int)fonts.size() && fonts[g_theme.font_idx])
      ImGui::PopFont();

    ImGui::Render();
    glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
    glClearColor(g_theme.bg.x, g_theme.bg.y, g_theme.bg.z, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(window);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();
  SDL_GL_DestroyContext(gl_context);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}
