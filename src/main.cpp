#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl3.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <climits>
#include <cmath>
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

#include "dct.h"
#include "entropy.h"
#include "loopFilter.h"
#include "quantization.h"

// =====================================================
// GLOBALS
// =====================================================
GLuint g_tex_original = 0;
GLuint g_tex_recon = 0;
GLuint g_tex_filtered = 0;
int g_tex_w = 0;
int g_tex_h = 0;

std::vector<unsigned char> g_gray;
std::vector<unsigned char> g_rgba_orig;
std::vector<DCTBlock> g_dct_blocks;

int g_block_size = 8;
int g_selected_block = -1;

// Quant
QuantMode g_quant_mode = QUANT_FLAT;
float g_base_q = 16.0f;
float g_deadzone_scale = 1.5f;
bool g_use_trellis = false;
float g_trellis_lambda = 0.5f;

// Entropy
EntropyModel g_entropy_model = ENTROPY_LAPLACIAN;
float g_fixed_sigma = 20.0f;
ContextModel g_ctx_model;

// Loop filter
LoopFilterParams g_lf_params;
bool g_lf_enabled = false;

// Results
bool g_experiment_run = false;
std::vector<QuantResult> g_quant_results;
std::vector<EntropyResult> g_entropy_results;
LoopFilterResult g_lf_result;
std::vector<unsigned char> g_recon_image;    // RGBA reconstructed
std::vector<unsigned char> g_filtered_image; // RGBA loop filtered

// Feature toggles
bool use_sign_hiding = false;
bool use_ccso = false;
bool use_gdf = false;
bool use_grain = false;
bool use_superblocks = false;

// =====================================================
// THEME STATE
// =====================================================
struct Theme {
  ImVec4 primary = ImVec4(0.36f, 0.71f, 0.31f, 1.0f);
  ImVec4 bg = ImVec4(0.08f, 0.08f, 0.08f, 1.0f);
  ImVec4 text = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  ImVec4 separator = ImVec4(0.30f, 0.30f, 0.30f, 1.0f);
  ImVec4 tab_bg = ImVec4(0.12f, 0.12f, 0.12f, 1.0f);
  float font_scale = 1.0f;
  int font_idx = 0; // index into loaded fonts
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
// HELPERS
// =====================================================
static float Clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
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

bool LoadImage(const char *filename) {
  int n;
  unsigned char *data = stbi_load(filename, &g_tex_w, &g_tex_h, &n, 4);
  if (!data)
    return false;
  g_rgba_orig.assign(data, data + g_tex_w * g_tex_h * 4);
  g_gray.resize(g_tex_w * g_tex_h);
  for (int i = 0; i < g_tex_w * g_tex_h; i++) {
    g_gray[i] =
        (unsigned char)(0.299f * data[i * 4] + 0.587f * data[i * 4 + 1] +
                        0.114f * data[i * 4 + 2]);
  }
  g_dct_blocks = ComputeDCTBlocks(g_gray, g_tex_w, g_tex_h, g_block_size);
  g_selected_block = -1;
  g_experiment_run = false;
  g_ctx_model.Init(g_block_size);
  if (g_tex_original)
    glDeleteTextures(1, &g_tex_original);
  g_tex_original = UploadTexture(data, g_tex_w, g_tex_h);
  stbi_image_free(data);
  return true;
}

void RunExperiment() {
  if (g_dct_blocks.empty())
    return;
  g_quant_results.clear();
  g_entropy_results.clear();
  g_recon_image.assign(g_tex_w * g_tex_h * 4, 255);
  g_ctx_model.Init(g_block_size);

  // Collect all coeff values for global sigma estimate
  std::vector<float> all_coeffs;
  for (auto &bl : g_dct_blocks)
    for (float v : bl.coeffs)
      all_coeffs.push_back(v);
  float global_sigma = EstimateLaplacianSigma(all_coeffs);

  for (auto &block : g_dct_blocks) {
    // Quantize
    QuantResult qres =
        QuantizeBlock(block, g_block_size, g_quant_mode, g_base_q,
                      g_deadzone_scale, g_use_trellis, g_trellis_lambda);
    qres.psnr = ComputePSNR(g_gray, qres.reconPixels, block.bx, block.by,
                            g_block_size, g_tex_w, g_tex_h);
    g_quant_results.push_back(qres);

    // Entropy
    EntropyResult eres =
        AnalyzeEntropy(qres, g_entropy_model, g_ctx_model, global_sigma);
    g_entropy_results.push_back(eres);

    // Paint recon pixels
    for (int r = 0; r < g_block_size; r++) {
      for (int c = 0; c < g_block_size; c++) {
        int ix = block.bx * g_block_size + c;
        int iy = block.by * g_block_size + r;
        if (ix >= g_tex_w || iy >= g_tex_h)
          continue;
        unsigned char pix = (unsigned char)Clampf(
            qres.reconPixels[r * g_block_size + c], 0, 255);
        int idx = (iy * g_tex_w + ix) * 4;
        g_recon_image[idx] = g_recon_image[idx + 1] = g_recon_image[idx + 2] =
            pix;
        g_recon_image[idx + 3] = 255;
      }
    }
  }

  if (g_tex_recon)
    UpdateTexture(g_tex_recon, g_recon_image.data(), g_tex_w, g_tex_h);
  else
    g_tex_recon = UploadTexture(g_recon_image.data(), g_tex_w, g_tex_h);

  // Loop filter pass
  if (g_lf_enabled) {
    std::vector<float> recon_f(g_tex_w * g_tex_h);
    for (int i = 0; i < g_tex_w * g_tex_h; i++)
      recon_f[i] = (float)g_recon_image[i * 4];
    g_lf_result =
        ApplyLoopFilter(recon_f, g_tex_w, g_tex_h, g_block_size, g_lf_params);
    g_lf_result.psnr_before = LFPSNR(g_gray, recon_f, g_tex_w, g_tex_h);
    g_lf_result.psnr_after =
        LFPSNR(g_gray, g_lf_result.filtered, g_tex_w, g_tex_h);

    g_filtered_image.resize(g_tex_w * g_tex_h * 4);
    for (int i = 0; i < g_tex_w * g_tex_h; i++) {
      unsigned char pix =
          (unsigned char)Clampf(g_lf_result.filtered[i], 0, 255);
      g_filtered_image[i * 4] = g_filtered_image[i * 4 + 1] =
          g_filtered_image[i * 4 + 2] = pix;
      g_filtered_image[i * 4 + 3] = 255;
    }
    if (g_tex_filtered)
      UpdateTexture(g_tex_filtered, g_filtered_image.data(), g_tex_w, g_tex_h);
    else
      g_tex_filtered = UploadTexture(g_filtered_image.data(), g_tex_w, g_tex_h);
  }

  g_experiment_run = true;
}

static ImU32 HeatColor(float t, unsigned char alpha = 180) {
  t = t < 0 ? 0 : (t > 1 ? 1 : t);
  float r, g, b;
  if (t < 0.25f) {
    float s = t / 0.25f;
    r = 0;
    g = s;
    b = 1;
  } else if (t < 0.5f) {
    float s = (t - 0.25f) / 0.25f;
    r = 0;
    g = 1;
    b = 1 - s;
  } else if (t < 0.75f) {
    float s = (t - 0.5f) / 0.25f;
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

  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();

  // Load fonts — add more .ttf paths here as needed
  ImFont *font_default = io.Fonts->AddFontDefault();
  ImFont *font_mono = nullptr;
  ImFont *font_large = nullptr;
  // Try to load system fonts; silently fall back if not present
  {
    static const char *mono_paths[] = {"C:/Windows/Fonts/consola.ttf",
                                       "C:/Windows/Fonts/cour.ttf", nullptr};
    for (int i = 0; mono_paths[i]; i++) {
      FILE *f = fopen(mono_paths[i], "rb");
      if (f) {
        fclose(f);
        font_mono = io.Fonts->AddFontFromFileTTF(mono_paths[i], 14.0f);
        break;
      }
    }
    static const char *sans_paths[] = {"C:/Windows/Fonts/segoeui.ttf",
                                       "C:/Windows/Fonts/arial.ttf", nullptr};
    for (int i = 0; sans_paths[i]; i++) {
      FILE *f = fopen(sans_paths[i], "rb");
      if (f) {
        fclose(f);
        font_large = io.Fonts->AddFontFromFileTTF(sans_paths[i], 15.0f);
        break;
      }
    }
  }

  // Collect available fonts for picker
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

  ImGui_ImplSDL3_InitForOpenGL(window, gl_context);
  ImGui_ImplOpenGL3_Init("#version 130");

  bool done = false;
  bool show_settings = false;
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

    // Apply active font
    if (g_theme.font_idx < (int)fonts.size() && fonts[g_theme.font_idx])
      ImGui::PushFont(fonts[g_theme.font_idx]);

    // ===== FULLSCREEN WINDOW =====
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::Begin("Codec Cooker", nullptr,
                 ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

    // ===== MENU =====
    if (ImGui::BeginMenuBar()) {
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Open Image")) {
          if (OpenFileDialog(file_path, sizeof(file_path)))
            LoadImage(file_path);
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

    // ===== THEME EDITOR =====
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

    // ===== LEFT PANEL =====
    ImGui::BeginChild("Left", ImVec2(280, 0), true);

    if (ImGui::Button("Open Image", ImVec2(-1, 32))) {
      if (OpenFileDialog(file_path, sizeof(file_path)))
        LoadImage(file_path);
    }

    ImGui::SeparatorText("Block");
    static int bs_idx = 1;
    const char *bs_opts[] = {"4x4", "8x8", "16x16", "32x32"};
    const int bs_vals[] = {4, 8, 16, 32};
    if (ImGui::Combo("Size##bs", &bs_idx, bs_opts, 4)) {
      g_block_size = bs_vals[bs_idx];
      if (!g_gray.empty()) {
        g_dct_blocks = ComputeDCTBlocks(g_gray, g_tex_w, g_tex_h, g_block_size);
        g_ctx_model.Init(g_block_size);
      }
    }

    ImGui::SeparatorText("Quantization");
    const char *qmode_names[] = {"Flat", "JPEG Perceptual", "Ramp", "Deadzone",
                                 "Custom"};
    int qm = (int)g_quant_mode;
    if (ImGui::Combo("Mode##qm", &qm, qmode_names, 5))
      g_quant_mode = (QuantMode)qm;
    ImGui::SliderFloat("Base Q", &g_base_q, 1.0f, 128.0f);
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
    ImGui::Checkbox("CCSO", &use_ccso);
    ImGui::SameLine();
    ImGui::Checkbox("GDF", &use_gdf);
    ImGui::Checkbox("Film Grain", &use_grain);
    ImGui::Checkbox("256 Superblocks", &use_superblocks);

    ImGui::Separator();
    bool can_run = !g_dct_blocks.empty();
    if (!can_run)
      ImGui::BeginDisabled();
    if (ImGui::Button("Run Experiment", ImVec2(-1, 40)))
      RunExperiment();
    if (!can_run)
      ImGui::EndDisabled();

    if (g_experiment_run && !g_quant_results.empty()) {
      ImGui::Separator();
      double total_bits_q = 0, total_bits_e = 0, total_bits_rle = 0;
      float avg_psnr = 0;
      int total_nz = 0;
      for (int i = 0; i < (int)g_quant_results.size(); i++) {
        total_bits_q += g_quant_results[i].estimatedBits;
        avg_psnr += g_quant_results[i].psnr;
        total_nz += g_quant_results[i].nonzeroCount;
        if (i < (int)g_entropy_results.size()) {
          total_bits_e += g_entropy_results[i].total_bits_model;
          total_bits_rle += g_entropy_results[i].total_bits_rle;
        }
      }
      avg_psnr /= (float)g_quant_results.size();
      ImGui::Text("Avg PSNR:    %.2f dB", avg_psnr);
      ImGui::Text("Quant bits:  %.0f", total_bits_q);
      ImGui::Text("Model bits:  %.0f", total_bits_e);
      ImGui::Text("RLE bits:    %.0f", total_bits_rle);
      ImGui::Text("Nonzero:     %d", total_nz);
      if (g_lf_enabled && g_experiment_run) {
        ImGui::Separator();
        ImGui::Text("LF PSNR before: %.2f dB", g_lf_result.psnr_before);
        ImGui::Text("LF PSNR after:  %.2f dB", g_lf_result.psnr_after);
        ImGui::Text("Avg edge str:   %.3f", g_lf_result.avg_boundary_strength);
      }
    }
    ImGui::EndChild();
    ImGui::SameLine();

    // ===== MAIN PANEL =====
    ImGui::BeginChild("Main", ImVec2(0, 0), true);
    if (ImGui::BeginTabBar("Tabs")) {

      // -------------------------------------------
      // TAB: Original
      // -------------------------------------------
      if (ImGui::BeginTabItem("Original")) {
        if (g_tex_original)
          ImGui::Image((ImTextureID)(intptr_t)g_tex_original,
                       ImVec2((float)g_tex_w, (float)g_tex_h));
        else
          ImGui::TextDisabled("No image loaded.");
        ImGui::EndTabItem();
      }

      // -------------------------------------------
      // TAB: Blocks
      // -------------------------------------------
      if (ImGui::BeginTabItem("Blocks")) {
        if (!g_tex_original) {
          ImGui::TextDisabled("Load an image first.");
        } else {
          float panel_w = ImGui::GetContentRegionAvail().x - 290;
          float scale = panel_w / (float)g_tex_w;
          float disp_w = g_tex_w * scale;
          float disp_h = g_tex_h * scale;
          ImVec2 img_pos = ImGui::GetCursorScreenPos();
          ImGui::Image((ImTextureID)(intptr_t)g_tex_original,
                       ImVec2(disp_w, disp_h));

          ImDrawList *draw = ImGui::GetWindowDrawList();
          float bs = g_block_size * scale;

          float max_energy = 1.0f;
          for (auto &bl : g_dct_blocks)
            if (bl.energy > max_energy)
              max_energy = bl.energy;

          for (int i = 0; i < (int)g_dct_blocks.size(); i++) {
            auto &bl = g_dct_blocks[i];
            float px = img_pos.x + bl.bx * bs;
            float py = img_pos.y + bl.by * bs;
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

          if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0)) {
            ImVec2 mouse = io.MousePos;
            int bx = (int)((mouse.x - img_pos.x) / bs);
            int by = (int)((mouse.y - img_pos.y) / bs);
            for (int i = 0; i < (int)g_dct_blocks.size(); i++)
              if (g_dct_blocks[i].bx == bx && g_dct_blocks[i].by == by) {
                g_selected_block = i;
                break;
              }
          }

          // Inspector
          ImGui::SameLine();
          ImGui::BeginChild("BlockInspector", ImVec2(280, 0), true);
          ImGui::Text("Block Inspector");
          ImGui::Separator();
          if (g_selected_block >= 0 &&
              g_selected_block < (int)g_dct_blocks.size()) {
            auto &bl = g_dct_blocks[g_selected_block];
            ImGui::Text("Block  (%d, %d)", bl.bx, bl.by);
            ImGui::Text("Energy %.2f", bl.energy);
            ImGui::Text("Origin (%d, %d)", bl.bx * g_block_size,
                        bl.by * g_block_size);
            if (g_experiment_run &&
                g_selected_block < (int)g_quant_results.size())
              ImGui::Text("PSNR   %.2f dB",
                          g_quant_results[g_selected_block].psnr);
            ImGui::Separator();
            ImGui::Text("DCT Coefficients:");
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
              int hc = (int)((io.MousePos.x - gpos.x) / cell);
              int hr = (int)((io.MousePos.y - gpos.y) / cell);
              if (hc >= 0 && hc < N && hr >= 0 && hr < N)
                ImGui::SetTooltip("[%d,%d] = %.4f", hr, hc,
                                  bl.coeffs[hr * N + hc]);
            }
          } else {
            ImGui::TextDisabled("Click a block to inspect.");
          }
          ImGui::EndChild();
        }
        ImGui::EndTabItem();
      }

      // -------------------------------------------
      // TAB: Slices
      // -------------------------------------------
      if (ImGui::BeginTabItem("Slices")) {
        if (!g_tex_original || g_gray.empty()) {
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

          // Helper: draw one slice plot
          auto DrawSlice = [&](bool horizontal, int line_idx,
                               float plot_h_val) {
            ImVec2 p0 = ImGui::GetCursorScreenPos();
            float W = avail.x - 4;
            float H = plot_h_val;
            int span = horizontal ? g_tex_w : g_tex_h;
            dl->AddRectFilled(p0, ImVec2(p0.x + W, p0.y + H),
                              IM_COL32(18, 18, 18, 255));

            auto getGray = [&](int i) -> float {
              int idx = horizontal ? (line_idx * g_tex_w + i)
                                   : (i * g_tex_w + line_idx);
              return (idx >= 0 && idx < (int)g_gray.size()) ? (float)g_gray[idx]
                                                            : 0;
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

            // Original (white)
            for (int x = 1; x < span; x++) {
              float v0 = getGray(x - 1), v1 = getGray(x);
              float px0 = p0.x + (x - 1) * W / span, px1 = p0.x + x * W / span;
              float py0 = p0.y + H - v0 * H / 255.0f,
                    py1 = p0.y + H - v1 * H / 255.0f;
              dl->AddLine(ImVec2(px0, py0), ImVec2(px1, py1),
                          IM_COL32(220, 220, 220, 255), 1.2f);
            }
            // Recon (green)
            if (show_recon && g_experiment_run) {
              for (int x = 1; x < span; x++) {
                float v0 = getRecon(x - 1), v1 = getRecon(x);
                if (v0 < 0 || v1 < 0)
                  break;
                float px0 = p0.x + (x - 1) * W / span,
                      px1 = p0.x + x * W / span;
                float py0 = p0.y + H - v0 * H / 255.0f,
                      py1 = p0.y + H - v1 * H / 255.0f;
                dl->AddLine(ImVec2(px0, py0), ImVec2(px1, py1),
                            IM_COL32(60, 210, 60, 200), 1.2f);
              }
            }
            // Filtered (orange)
            if (show_filtered && g_lf_enabled && g_experiment_run) {
              for (int x = 1; x < span; x++) {
                float v0 = getFiltered(x - 1), v1 = getFiltered(x);
                if (v0 < 0 || v1 < 0)
                  break;
                float px0 = p0.x + (x - 1) * W / span,
                      px1 = p0.x + x * W / span;
                float py0 = p0.y + H - v0 * H / 255.0f,
                      py1 = p0.y + H - v1 * H / 255.0f;
                dl->AddLine(ImVec2(px0, py0), ImVec2(px1, py1),
                            IM_COL32(255, 140, 30, 200), 1.2f);
              }
            }
            // Block boundaries (blue verticals)
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

      // -------------------------------------------
      // TAB: Quantization
      // -------------------------------------------
      if (ImGui::BeginTabItem("Quantization")) {
        if (!g_experiment_run) {
          ImGui::TextDisabled("Run an experiment first.");
        } else if (g_selected_block < 0 ||
                   g_selected_block >= (int)g_quant_results.size()) {
          ImGui::TextDisabled("Click a block in the Blocks tab.");
        } else {
          QuantResult &res = g_quant_results[g_selected_block];
          DCTBlock &blk = g_dct_blocks[g_selected_block];
          int N = g_block_size;
          ImGui::Text(
              "Block (%d,%d)  PSNR: %.2f dB  Nonzero: %d/%d  Est bits: %.1f",
              blk.bx, blk.by, res.psnr, res.nonzeroCount, N * N,
              res.estimatedBits);
          ImGui::Separator();

          // Three pixel patches
          ImVec2 patch_pos = ImGui::GetCursorScreenPos();
          float cell = 12.0f;
          float patch_size = N * cell;
          ImDrawList *dl = ImGui::GetWindowDrawList();

          for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) {
              int ix = blk.bx * N + c, iy = blk.by * N + r;
              unsigned char pv = (ix < g_tex_w && iy < g_tex_h)
                                     ? g_gray[iy * g_tex_w + ix]
                                     : 0;
              ImVec2 p0 =
                  ImVec2(patch_pos.x + c * cell, patch_pos.y + r * cell);
              dl->AddRectFilled(p0, ImVec2(p0.x + cell - 1, p0.y + cell - 1),
                                IM_COL32(pv, pv, pv, 255));
            }
          float rx = patch_size + 16;
          for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) {
              unsigned char pv =
                  (unsigned char)Clampf(res.reconPixels[r * N + c], 0, 255);
              ImVec2 p0 =
                  ImVec2(patch_pos.x + rx + c * cell, patch_pos.y + r * cell);
              dl->AddRectFilled(p0, ImVec2(p0.x + cell - 1, p0.y + cell - 1),
                                IM_COL32(pv, pv, pv, 255));
            }
          float ex = rx + patch_size + 16;
          for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) {
              int ix = blk.bx * N + c, iy = blk.by * N + r;
              float orig_pix = (ix < g_tex_w && iy < g_tex_h)
                                   ? (float)g_gray[iy * g_tex_w + ix]
                                   : 0;
              float err =
                  fabsf(orig_pix - Clampf(res.reconPixels[r * N + c], 0, 255));
              ImVec2 p0 =
                  ImVec2(patch_pos.x + ex + c * cell, patch_pos.y + r * cell);
              dl->AddRectFilled(p0, ImVec2(p0.x + cell - 1, p0.y + cell - 1),
                                HeatColor(err / 64.0f, 255));
            }
          ImGui::Dummy(ImVec2(1, patch_size + 6));
          ImGui::Text("Original");
          ImGui::SameLine(patch_size + 16);
          ImGui::Text("Reconstructed");
          ImGui::SameLine(rx + patch_size + 16);
          ImGui::Text("Error x4 (heat)");
          ImGui::Separator();

          // Coefficient table
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

      // -------------------------------------------
      // TAB: Entropy
      // -------------------------------------------
      if (ImGui::BeginTabItem("Entropy")) {
        if (!g_experiment_run) {
          ImGui::TextDisabled("Run an experiment first.");
        } else if (g_selected_block < 0 ||
                   g_selected_block >= (int)g_entropy_results.size()) {
          ImGui::TextDisabled("Click a block in the Blocks tab.");
        } else {
          EntropyResult &er = g_entropy_results[g_selected_block];
          DCTBlock &bl = g_dct_blocks[g_selected_block];
          int N = g_block_size;

          ImGui::Text("Block (%d,%d)  Nonzero: %d/%d", bl.bx, bl.by,
                      er.nonzero_count, N * N);
          ImGui::Text("Naive bits: %.1f   Model bits: %.1f   RLE bits: %.1f",
                      er.total_bits_naive, er.total_bits_model,
                      er.total_bits_rle);

          float saving =
              er.total_bits_naive > 0
                  ? (float)(1.0 - er.total_bits_model / er.total_bits_naive) *
                        100.0f
                  : 0;
          ImGui::TextColored(ImVec4(0.4f, 1, 0.4f, 1),
                             "Model saves %.1f%% vs naive", saving);
          ImGui::Separator();

          // Bit cost heatmap
          ImGui::Text("Bit cost per coefficient (hotter = more expensive):");
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
              // Draw nonzero marker
              if (er.entries[r * N + c].level != 0)
                dl->AddRect(p0, ImVec2(p0.x + cell - 1, p0.y + cell - 1),
                            IM_COL32(255, 255, 255, 180), 0, 0, 1.0f);
            }
          if (ImGui::IsMouseHoveringRect(gpos,
                                         ImVec2(gpos.x + 252, gpos.y + 252))) {
            int hc = (int)((io.MousePos.x - gpos.x) / cell);
            int hr = (int)((io.MousePos.y - gpos.y) / cell);
            if (hc >= 0 && hc < N && hr >= 0 && hr < N) {
              auto &en = er.entries[hr * N + hc];
              ImGui::SetTooltip("[%d,%d]  level=%d  prob=%.5f  bits=%.2f", hr,
                                hc, en.level, en.probability, en.bit_cost);
            }
          }
          ImGui::Dummy(ImVec2(252, 252));
          ImGui::Text("White border = nonzero coefficient");

          ImGui::Separator();

          // RLE sequence
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

          // Per-coeff entropy table
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
              ImGui::TextColored(HeatColor(en.bit_cost / 20.0f) == 0
                                     ? ImVec4(1, 1, 1, 1)
                                     : ImVec4(1, 0.6f, 0.3f, 1),
                                 "%.2f", en.bit_cost);
              ImGui::TableSetColumnIndex(4);
              ImGui::Text("%.1f", en.laplace_sigma);
            }
            ImGui::EndTable();
          }
        }
        ImGui::EndTabItem();
      }

      // -------------------------------------------
      // TAB: Loop Filter
      // -------------------------------------------
      if (ImGui::BeginTabItem("Loop Filter")) {
        if (!g_experiment_run || !g_lf_enabled) {
          ImGui::TextDisabled(
              "Enable Loop Filter in left panel, then Run Experiment.");
        } else {
          ImGui::Text("PSNR before filter: %.2f dB", g_lf_result.psnr_before);
          ImGui::SameLine(0, 20);
          float gain = g_lf_result.psnr_after - g_lf_result.psnr_before;
          ImGui::TextColored(
              gain > 0 ? ImVec4(0.4f, 1, 0.4f, 1) : ImVec4(1, 0.4f, 0.4f, 1),
              "After: %.2f dB (%+.2f dB)", g_lf_result.psnr_after, gain);
          ImGui::Text("Avg boundary strength: %.4f",
                      g_lf_result.avg_boundary_strength);
          ImGui::Separator();

          // Side by side: reconstructed | filtered
          float half = (ImGui::GetContentRegionAvail().x - 8) / 2.0f;
          float sc = half / (float)g_tex_w;
          float dw = g_tex_w * sc, dh = g_tex_h * sc;
          ImGui::Text("Reconstructed (pre-filter)");
          ImGui::SameLine(half + 8);
          ImGui::Text("Loop Filtered");
          if (g_tex_recon)
            ImGui::Image((ImTextureID)(intptr_t)g_tex_recon, ImVec2(dw, dh));
          ImGui::SameLine();
          if (g_tex_filtered)
            ImGui::Image((ImTextureID)(intptr_t)g_tex_filtered, ImVec2(dw, dh));

          // Boundary strength heatmap overlay
          ImGui::Separator();
          ImGui::Text("Boundary strength map (hover for value):");
          ImVec2 map_pos = ImGui::GetCursorScreenPos();
          float map_scale = half / g_tex_w;
          float bsz = g_block_size * map_scale;
          ImDrawList *dl = ImGui::GetWindowDrawList();

          // Draw gray bg
          dl->AddRectFilled(map_pos,
                            ImVec2(map_pos.x + g_tex_w * map_scale,
                                   map_pos.y + g_tex_h * map_scale),
                            IM_COL32(30, 30, 30, 255));

          for (auto &bd : g_lf_result.boundaries) {
            if (bd.horizontal) {
              float x0 = map_pos.x + bd.bx * bsz;
              float y0 = map_pos.y + (bd.by + 1) * bsz;
              dl->AddLine(ImVec2(x0, y0), ImVec2(x0 + bsz, y0),
                          HeatColor(bd.strength, 255), 2.0f);
            } else {
              float x0 = map_pos.x + (bd.bx + 1) * bsz;
              float y0 = map_pos.y + bd.by * bsz;
              dl->AddLine(ImVec2(x0, y0), ImVec2(x0, y0 + bsz),
                          HeatColor(bd.strength, 255), 2.0f);
            }
          }
          ImGui::Dummy(ImVec2(g_tex_w * map_scale, g_tex_h * map_scale));
        }
        ImGui::EndTabItem();
      }

      // -------------------------------------------
      // TAB: Reconstructed
      // -------------------------------------------
      if (ImGui::BeginTabItem("Reconstructed")) {
        if (!g_experiment_run) {
          ImGui::TextDisabled("Run an experiment first.");
        } else {
          float half = (ImGui::GetContentRegionAvail().x - 12) / 2.0f;
          float sc = half / (float)g_tex_w;
          float dw = g_tex_w * sc, dh = g_tex_h * sc;
          const char *mnames[] = {"Flat", "JPEG", "Ramp", "Deadzone", "Custom"};
          ImGui::Text("Original");
          ImGui::SameLine(half + 12);
          ImGui::Text("Reconstructed  Q=%.0f  %s", g_base_q,
                      mnames[(int)g_quant_mode]);
          ImGui::Image((ImTextureID)(intptr_t)g_tex_original, ImVec2(dw, dh));
          ImGui::SameLine();
          if (g_tex_recon)
            ImGui::Image((ImTextureID)(intptr_t)g_tex_recon, ImVec2(dw, dh));
        }
        ImGui::EndTabItem();
      }

      // -------------------------------------------
      // TAB: Heatmaps
      // -------------------------------------------
      if (ImGui::BeginTabItem("Heatmaps")) {
        if (g_dct_blocks.empty()) {
          ImGui::TextDisabled("Load an image first.");
        } else {
          ImGui::Text("DCT Energy per block (brighter = higher energy)");
          float pw = ImGui::GetContentRegionAvail().x;
          float sc = pw / (float)g_tex_w;
          float bs = g_block_size * sc;
          ImVec2 base = ImGui::GetCursorScreenPos();
          ImGui::Dummy(ImVec2(g_tex_w * sc, g_tex_h * sc));
          ImDrawList *dl = ImGui::GetWindowDrawList();
          float mx = 1.0f;
          for (auto &bl : g_dct_blocks)
            if (bl.energy > mx)
              mx = bl.energy;
          for (auto &bl : g_dct_blocks) {
            float px = base.x + bl.bx * bs, py = base.y + bl.by * bs;
            dl->AddRectFilled(ImVec2(px, py), ImVec2(px + bs, py + bs),
                              HeatColor(bl.energy / mx, 230));
            dl->AddRect(ImVec2(px, py), ImVec2(px + bs, py + bs),
                        IM_COL32(0, 0, 0, 60));
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
