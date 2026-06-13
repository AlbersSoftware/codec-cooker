#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl3.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL3/SDL_opengles2.h>
#else
#include <SDL3/SDL_opengl.h>
#endif

#define WIN32_LEAN_AND_MEAN
#include <commdlg.h>
#include <windows.h>

#include "dct.h"
#include "quantization.h"

// ===== GLOBALS =====
GLuint g_tex_original = 0;
GLuint g_tex_recon = 0;
int g_tex_w = 0;
int g_tex_h = 0;

std::vector<unsigned char> g_gray;
std::vector<unsigned char> g_rgba_orig;
std::vector<DCTBlock> g_dct_blocks;

int g_block_size = 8;
int g_selected_block = -1;

// Quant experiment state
QuantMode g_quant_mode = QUANT_FLAT;
float g_base_q = 16.0f;
float g_deadzone_scale = 1.5f;
bool g_use_trellis = false;
float g_trellis_lambda = 0.5f;
bool g_experiment_run = false;

std::vector<QuantResult> g_quant_results;
std::vector<unsigned char> g_recon_image; // full reconstructed image RGBA

// Feature toggles
bool use_sign_hiding = false;
bool use_ccso = false;
bool use_gdf = false;
bool use_grain = false;
bool use_superblocks = false;

// ===== HELPERS =====
static float Clamp(float v, float lo, float hi) {
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

  // Grayscale
  g_gray.resize(g_tex_w * g_tex_h);
  for (int i = 0; i < g_tex_w * g_tex_h; i++) {
    int r = data[i * 4 + 0], gv = data[i * 4 + 1], b = data[i * 4 + 2];
    g_gray[i] = (unsigned char)(0.299f * r + 0.587f * gv + 0.114f * b);
  }

  // DCT
  g_dct_blocks = ComputeDCTBlocks(g_gray, g_tex_w, g_tex_h, g_block_size);
  g_selected_block = -1;
  g_experiment_run = false;

  if (g_tex_original)
    glDeleteTextures(1, &g_tex_original);
  g_tex_original = UploadTexture(data, g_tex_w, g_tex_h);

  stbi_image_free(data);
  return true;
}

// Run quantization on every block, rebuild reconstructed RGBA texture
void RunExperiment() {
  if (g_dct_blocks.empty())
    return;

  g_quant_results.clear();
  g_recon_image.assign(g_tex_w * g_tex_h * 4, 255);

  for (auto &block : g_dct_blocks) {
    QuantResult res =
        QuantizeBlock(block, g_block_size, g_quant_mode, g_base_q,
                      g_deadzone_scale, g_use_trellis, g_trellis_lambda);

    // Compute PSNR
    res.psnr = ComputePSNR(g_gray, res.reconPixels, block.bx, block.by,
                           g_block_size, g_tex_w, g_tex_h);

    g_quant_results.push_back(res);

    // Paint reconstructed pixels into image buffer
    for (int r = 0; r < g_block_size; r++) {
      for (int c = 0; c < g_block_size; c++) {
        int ix = block.bx * g_block_size + c;
        int iy = block.by * g_block_size + r;
        if (ix >= g_tex_w || iy >= g_tex_h)
          continue;
        unsigned char pix = (unsigned char)Clamp(
            res.reconPixels[r * g_block_size + c], 0.0f, 255.0f);
        int idx = (iy * g_tex_w + ix) * 4;
        g_recon_image[idx + 0] = pix;
        g_recon_image[idx + 1] = pix;
        g_recon_image[idx + 2] = pix;
        g_recon_image[idx + 3] = 255;
      }
    }
  }

  if (g_tex_recon) {
    UpdateTexture(g_tex_recon, g_recon_image.data(), g_tex_w, g_tex_h);
  } else {
    g_tex_recon = UploadTexture(g_recon_image.data(), g_tex_w, g_tex_h);
  }

  g_experiment_run = true;
}

// Heat color: blue->cyan->green->yellow->red
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

// ===== MAIN =====
int main(int, char **) {
  SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

  SDL_Window *window = SDL_CreateWindow(
      "Codec Workbench", 1600, 1000,
      SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);

  SDL_GLContext gl_context = SDL_GL_CreateContext(window);
  SDL_GL_MakeCurrent(window, gl_context);
  SDL_GL_SetSwapInterval(1);

  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  ImGui::StyleColorsDark();
  ImGuiStyle &s = ImGui::GetStyle();

  ImVec4 primary = ImVec4(0.36f, 0.71f, 0.31f, 1.0f);
  ImVec4 bg = ImVec4(0.08f, 0.08f, 0.08f, 1.0f);

  s.Colors[ImGuiCol_WindowBg] = bg;
  s.Colors[ImGuiCol_MenuBarBg] = bg;
  s.Colors[ImGuiCol_TitleBg] = bg;
  s.Colors[ImGuiCol_TitleBgActive] = bg;
  s.Colors[ImGuiCol_ChildBg] = bg;
  s.Colors[ImGuiCol_Button] = primary;
  s.Colors[ImGuiCol_Header] = primary;
  s.Colors[ImGuiCol_Tab] = ImVec4(0.12f, 0.12f, 0.12f, 1);
  s.Colors[ImGuiCol_TabActive] = primary;

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

    // ===== FULL SCREEN WINDOW =====
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::Begin("Codec Workbench", nullptr,
                 ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

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

    // ===== SETTINGS POPUP =====
    if (show_settings) {
      ImGui::SetNextWindowSize(ImVec2(420, 320), ImGuiCond_FirstUseEver);
      ImGui::Begin("Theme Settings", &show_settings);
      float p[3] = {primary.x, primary.y, primary.z};
      float b[3] = {bg.x, bg.y, bg.z};
      if (ImGui::ColorEdit3("Primary", p)) {
        primary = ImVec4(p[0], p[1], p[2], 1);
        s.Colors[ImGuiCol_Button] = primary;
        s.Colors[ImGuiCol_Header] = primary;
        s.Colors[ImGuiCol_TabActive] = primary;
      }
      if (ImGui::ColorEdit3("Background", b)) {
        bg = ImVec4(b[0], b[1], b[2], 1);
        s.Colors[ImGuiCol_WindowBg] = bg;
        s.Colors[ImGuiCol_MenuBarBg] = bg;
        s.Colors[ImGuiCol_ChildBg] = bg;
      }
      static float font_scale = 1.0f;
      ImGui::SliderFloat("Font Scale", &font_scale, 0.5f, 2.0f);
      io.FontGlobalScale = font_scale;
      ImGui::End();
    }

    // ===== LEFT PANEL =====
    ImGui::BeginChild("Left", ImVec2(270, 0), true);

    if (ImGui::Button("Open Image", ImVec2(-1, 32))) {
      if (OpenFileDialog(file_path, sizeof(file_path)))
        LoadImage(file_path);
    }

    ImGui::Separator();
    ImGui::Text("Block Size");
    static int bs_idx = 1; // default 8x8
    const char *bs_opts[] = {"4x4", "8x8", "16x16", "32x32"};
    const int bs_vals[] = {4, 8, 16, 32};
    if (ImGui::Combo("##bs", &bs_idx, bs_opts, 4)) {
      g_block_size = bs_vals[bs_idx];
      if (!g_gray.empty())
        g_dct_blocks = ComputeDCTBlocks(g_gray, g_tex_w, g_tex_h, g_block_size);
    }

    ImGui::Separator();
    ImGui::Text("Quantization");

    const char *mode_names[] = {"Flat", "JPEG Perceptual", "Ramp", "Deadzone",
                                "Custom"};
    int qm = (int)g_quant_mode;
    if (ImGui::Combo("Mode##qm", &qm, mode_names, 5))
      g_quant_mode = (QuantMode)qm;

    ImGui::SliderFloat("Base Q", &g_base_q, 1.0f, 128.0f);

    if (g_quant_mode == QUANT_DEADZONE)
      ImGui::SliderFloat("Deadzone Scale", &g_deadzone_scale, 1.0f, 4.0f);

    ImGui::Checkbox("Trellis (RDO)", &g_use_trellis);
    if (g_use_trellis)
      ImGui::SliderFloat("Lambda", &g_trellis_lambda, 0.01f, 5.0f);

    ImGui::Separator();
    ImGui::Text("Entropy");
    ImGui::Checkbox("Sign Hiding", &use_sign_hiding);

    ImGui::Text("Post Processing");
    ImGui::Checkbox("CCSO", &use_ccso);
    ImGui::SameLine();
    ImGui::Checkbox("GDF", &use_gdf);
    ImGui::Checkbox("Film Grain", &use_grain);

    ImGui::Text("Structure");
    ImGui::Checkbox("256x256 Superblocks", &use_superblocks);

    ImGui::Separator();

    bool can_run = !g_dct_blocks.empty();
    if (!can_run)
      ImGui::BeginDisabled();
    if (ImGui::Button("Run Experiment", ImVec2(-1, 40)))
      RunExperiment();
    if (!can_run)
      ImGui::EndDisabled();

    // Summary stats after run
    if (g_experiment_run && !g_quant_results.empty()) {
      ImGui::Separator();
      double total_bits = 0;
      float avg_psnr = 0;
      int total_nz = 0;
      for (auto &r : g_quant_results) {
        total_bits += r.estimatedBits;
        avg_psnr += r.psnr;
        total_nz += r.nonzeroCount;
      }
      avg_psnr /= (float)g_quant_results.size();
      ImGui::Text("Avg PSNR:  %.2f dB", avg_psnr);
      ImGui::Text("Est. bits: %.0f", total_bits);
      ImGui::Text("Nonzero:   %d", total_nz);
    }

    ImGui::EndChild();
    ImGui::SameLine();

    // ===== MAIN PANEL =====
    ImGui::BeginChild("Main", ImVec2(0, 0), true);

    if (ImGui::BeginTabBar("Tabs")) {

      // ------------------------------------------------
      // TAB: Original
      // ------------------------------------------------
      if (ImGui::BeginTabItem("Original")) {
        if (g_tex_original)
          ImGui::Image((ImTextureID)(intptr_t)g_tex_original,
                       ImVec2((float)g_tex_w, (float)g_tex_h));
        else
          ImGui::TextDisabled(
              "No image loaded. Use 'Open Image' or File menu.");
        ImGui::EndTabItem();
      }

      // ------------------------------------------------
      // TAB: Blocks — overlay with grid + inspector
      // ------------------------------------------------
      if (ImGui::BeginTabItem("Blocks")) {
        if (!g_tex_original) {
          ImGui::TextDisabled("Load an image first.");
        } else {
          // Scale image to fit panel width
          float panel_w = ImGui::GetContentRegionAvail().x - 280;
          float scale = panel_w / (float)g_tex_w;
          float disp_w = g_tex_w * scale;
          float disp_h = g_tex_h * scale;

          ImVec2 img_pos = ImGui::GetCursorScreenPos();
          ImGui::Image((ImTextureID)(intptr_t)g_tex_original,
                       ImVec2(disp_w, disp_h));

          ImDrawList *draw = ImGui::GetWindowDrawList();
          float bs = g_block_size * scale;

          // Energy for heatmap normalization
          float max_energy = 1.0f;
          for (auto &bl : g_dct_blocks)
            if (bl.energy > max_energy)
              max_energy = bl.energy;

          // Draw block overlays
          int blocks_x = (g_tex_w + g_block_size - 1) / g_block_size;
          int blocks_y = (g_tex_h + g_block_size - 1) / g_block_size;

          for (int i = 0; i < (int)g_dct_blocks.size(); i++) {
            auto &bl = g_dct_blocks[i];
            float px = img_pos.x + bl.bx * bs;
            float py = img_pos.y + bl.by * bs;

            // Heat fill
            float norm = bl.energy / max_energy;
            ImU32 fill = (i == g_selected_block) ? IM_COL32(255, 255, 0, 120)
                                                 : HeatColor(norm, 60);
            draw->AddRectFilled(ImVec2(px, py),
                                ImVec2(px + bs - 1, py + bs - 1), fill);

            // Always-visible block border
            ImU32 border = (i == g_selected_block)
                               ? IM_COL32(255, 255, 0, 255)
                               : IM_COL32(200, 200, 200, 80);
            draw->AddRect(ImVec2(px, py), ImVec2(px + bs, py + bs), border,
                          0.0f, 0, 1.0f);
          }

          // Click to select block
          if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0)) {
            ImVec2 mouse = io.MousePos;
            int bx = (int)((mouse.x - img_pos.x) / bs);
            int by = (int)((mouse.y - img_pos.y) / bs);
            for (int i = 0; i < (int)g_dct_blocks.size(); i++) {
              if (g_dct_blocks[i].bx == bx && g_dct_blocks[i].by == by) {
                g_selected_block = i;
                break;
              }
            }
          }

          // Right panel: block inspector
          ImGui::SameLine();
          ImGui::BeginChild("BlockInspector", ImVec2(270, 0), true);
          ImGui::Text("Block Inspector");
          ImGui::Separator();

          if (g_selected_block >= 0 &&
              g_selected_block < (int)g_dct_blocks.size()) {
            auto &bl = g_dct_blocks[g_selected_block];
            ImGui::Text("Block  (%d, %d)", bl.bx, bl.by);
            ImGui::Text("Energy %.2f", bl.energy);
            ImGui::Text("Pixel  (%d, %d)", bl.bx * g_block_size,
                        bl.by * g_block_size);
            ImGui::Separator();
            ImGui::Text("DCT Coefficients:");

            // Draw coefficient grid
            int N = g_block_size;
            float cell = 220.0f / N;
            ImVec2 grid_pos = ImGui::GetCursorScreenPos();

            float max_c = 0.001f;
            for (auto &v : bl.coeffs)
              if (fabsf(v) > max_c)
                max_c = fabsf(v);

            for (int r = 0; r < N; r++) {
              for (int c = 0; c < N; c++) {
                float val = bl.coeffs[r * N + c];
                float t = fabsf(val) / max_c;
                ImU32 col = HeatColor(t, 220);
                ImVec2 p0 =
                    ImVec2(grid_pos.x + c * cell, grid_pos.y + r * cell);
                ImVec2 p1 = ImVec2(p0.x + cell - 1, p0.y + cell - 1);
                draw->AddRectFilled(p0, p1, col);
              }
            }
            ImGui::Dummy(ImVec2(220, 220));

            // Tooltip on hover over grid
            if (ImGui::IsMouseHoveringRect(
                    grid_pos, ImVec2(grid_pos.x + 220, grid_pos.y + 220))) {
              int hc = (int)((io.MousePos.x - grid_pos.x) / cell);
              int hr = (int)((io.MousePos.y - grid_pos.y) / cell);
              if (hc >= 0 && hc < N && hr >= 0 && hr < N) {
                float cv = bl.coeffs[hr * N + hc];
                ImGui::SetTooltip("[%d,%d] = %.4f", hr, hc, cv);
              }
            }
          } else {
            ImGui::TextDisabled("Click a block on the image.");
          }
          ImGui::EndChild();
        }
        ImGui::EndTabItem();
      }

      // ------------------------------------------------
      // TAB: Slices — horizontal + vertical luminance slices
      // ------------------------------------------------
      if (ImGui::BeginTabItem("Slices")) {
        if (!g_tex_original || g_gray.empty()) {
          ImGui::TextDisabled("Load an image first.");
        } else {
          static int slice_row = 0;
          static int slice_col = 0;
          ImGui::SliderInt("Row slice (Y)", &slice_row, 0, g_tex_h - 1);
          ImGui::SliderInt("Col slice (X)", &slice_col, 0, g_tex_w - 1);

          ImVec2 avail = ImGui::GetContentRegionAvail();
          float plot_h = (avail.y - 80) / 2.0f;

          // --- Horizontal slice (across a row) ---
          ImGui::Separator();
          ImGui::Text("Horizontal slice — row %d", slice_row);
          {
            ImVec2 p0 = ImGui::GetCursorScreenPos();
            float W = avail.x - 4;
            float H = plot_h;
          draw_list_helper:;
            ImDrawList *dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(p0, ImVec2(p0.x + W, p0.y + H),
                              IM_COL32(20, 20, 20, 255));

            // Original (white)
            for (int x = 1; x < g_tex_w; x++) {
              float v0 = g_gray[(slice_row * g_tex_w + (x - 1))];
              float v1 = g_gray[(slice_row * g_tex_w + x)];
              float px0 = p0.x + (x - 1) * W / g_tex_w;
              float px1 = p0.x + x * W / g_tex_w;
              float py0 = p0.y + H - v0 * H / 255.0f;
              float py1 = p0.y + H - v1 * H / 255.0f;
              dl->AddLine(ImVec2(px0, py0), ImVec2(px1, py1),
                          IM_COL32(220, 220, 220, 255), 1.0f);
            }

            // Reconstructed (green) — only if experiment was run
            if (g_experiment_run && !g_recon_image.empty()) {
              for (int x = 1; x < g_tex_w; x++) {
                int idx0 = (slice_row * g_tex_w + (x - 1)) * 4;
                int idx1 = (slice_row * g_tex_w + x) * 4;
                float v0 = g_recon_image[idx0];
                float v1 = g_recon_image[idx1];
                float px0 = p0.x + (x - 1) * W / g_tex_w;
                float px1 = p0.x + x * W / g_tex_w;
                float py0 = p0.y + H - v0 * H / 255.0f;
                float py1 = p0.y + H - v1 * H / 255.0f;
                dl->AddLine(ImVec2(px0, py0), ImVec2(px1, py1),
                            IM_COL32(80, 200, 80, 200), 1.0f);
              }
            }

            // Block boundary lines
            for (int bx = 0; bx < g_tex_w; bx += g_block_size) {
              float lx = p0.x + bx * W / g_tex_w;
              ImGui::GetWindowDrawList()->AddLine(
                  ImVec2(lx, p0.y), ImVec2(lx, p0.y + H),
                  IM_COL32(80, 80, 200, 100), 1.0f);
            }

            ImGui::Dummy(ImVec2(W, H));
            ImGui::Text("White = original   Green = reconstructed   Blue lines "
                        "= block boundaries");
          }

          // --- Vertical slice (down a column) ---
          ImGui::Separator();
          ImGui::Text("Vertical slice — col %d", slice_col);
          {
            ImVec2 p0 = ImGui::GetCursorScreenPos();
            float W = avail.x - 4;
            float H = plot_h;
            ImDrawList *dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(p0, ImVec2(p0.x + W, p0.y + H),
                              IM_COL32(20, 20, 20, 255));

            for (int y = 1; y < g_tex_h; y++) {
              float v0 = g_gray[((y - 1) * g_tex_w + slice_col)];
              float v1 = g_gray[(y * g_tex_w + slice_col)];
              float px0 = p0.x + (y - 1) * W / g_tex_h;
              float px1 = p0.x + y * W / g_tex_h;
              float py0 = p0.y + H - v0 * H / 255.0f;
              float py1 = p0.y + H - v1 * H / 255.0f;
              dl->AddLine(ImVec2(px0, py0), ImVec2(px1, py1),
                          IM_COL32(220, 220, 220, 255), 1.0f);
            }

            if (g_experiment_run && !g_recon_image.empty()) {
              for (int y = 1; y < g_tex_h; y++) {
                int idx0 = ((y - 1) * g_tex_w + slice_col) * 4;
                int idx1 = (y * g_tex_w + slice_col) * 4;
                float v0 = g_recon_image[idx0];
                float v1 = g_recon_image[idx1];
                float px0 = p0.x + (y - 1) * W / g_tex_h;
                float px1 = p0.x + y * W / g_tex_h;
                float py0 = p0.y + H - v0 * H / 255.0f;
                float py1 = p0.y + H - v1 * H / 255.0f;
                dl->AddLine(ImVec2(px0, py0), ImVec2(px1, py1),
                            IM_COL32(80, 200, 80, 200), 1.0f);
              }
            }

            for (int by = 0; by < g_tex_h; by += g_block_size) {
              float ly = p0.x + by * W / g_tex_h;
              dl->AddLine(ImVec2(ly, p0.y), ImVec2(ly, p0.y + H),
                          IM_COL32(80, 80, 200, 100), 1.0f);
            }

            ImGui::Dummy(ImVec2(W, H));
          }
        }
        ImGui::EndTabItem();
      }

      // ------------------------------------------------
      // TAB: Quantization — per-coefficient breakdown
      // ------------------------------------------------
      if (ImGui::BeginTabItem("Quantization")) {
        if (!g_experiment_run) {
          ImGui::TextDisabled("Run an experiment first (left panel).");
        } else if (g_selected_block < 0 ||
                   g_selected_block >= (int)g_quant_results.size()) {
          ImGui::TextDisabled("Click a block in the Blocks tab to inspect it.");
        } else {
          QuantResult &res = g_quant_results[g_selected_block];
          DCTBlock &blk = g_dct_blocks[g_selected_block];
          int N = g_block_size;

          ImGui::Text("Block (%d, %d)  |  PSNR: %.2f dB  |  "
                      "Nonzero: %d/%d  |  Est bits: %.1f",
                      blk.bx, blk.by, res.psnr, res.nonzeroCount, N * N,
                      res.estimatedBits);

          ImGui::Separator();

          // Side by side: original patch | reconstructed patch
          ImVec2 patch_pos = ImGui::GetCursorScreenPos();
          float cell = 12.0f;
          float patch_size = N * cell;
          ImDrawList *dl = ImGui::GetWindowDrawList();

          // Original pixels (grayscale)
          for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
              int ix = blk.bx * N + c;
              int iy = blk.by * N + r;
              unsigned char pv = (ix < g_tex_w && iy < g_tex_h)
                                     ? g_gray[iy * g_tex_w + ix]
                                     : 0;
              ImVec2 p0 =
                  ImVec2(patch_pos.x + c * cell, patch_pos.y + r * cell);
              dl->AddRectFilled(p0, ImVec2(p0.x + cell - 1, p0.y + cell - 1),
                                IM_COL32(pv, pv, pv, 255));
            }
          }

          // Reconstructed pixels
          float rx_off = patch_size + 20;
          for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
              float rv = res.reconPixels[r * N + c];
              unsigned char pv = (unsigned char)Clamp(rv, 0, 255);
              ImVec2 p0 = ImVec2(patch_pos.x + rx_off + c * cell,
                                 patch_pos.y + r * cell);
              dl->AddRectFilled(p0, ImVec2(p0.x + cell - 1, p0.y + cell - 1),
                                IM_COL32(pv, pv, pv, 255));
            }
          }

          // Error heatmap (x4 amplified)
          float ex_off = rx_off + patch_size + 20;
          for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
              int ix = blk.bx * N + c;
              int iy = blk.by * N + r;
              float orig_pix = (ix < g_tex_w && iy < g_tex_h)
                                   ? (float)g_gray[iy * g_tex_w + ix]
                                   : 0;
              float err =
                  fabsf(orig_pix - Clamp(res.reconPixels[r * N + c], 0, 255));
              ImVec2 p0 = ImVec2(patch_pos.x + ex_off + c * cell,
                                 patch_pos.y + r * cell);
              dl->AddRectFilled(p0, ImVec2(p0.x + cell - 1, p0.y + cell - 1),
                                HeatColor(err / 64.0f, 255));
            }
          }

          // Labels
          ImGui::Dummy(ImVec2(1, patch_size + 6));
          ImGui::Text("Original");
          ImGui::SameLine(patch_size + 20);
          ImGui::Text("Reconstructed");
          ImGui::SameLine(rx_off + patch_size + 20);
          ImGui::Text("Error (4x)");

          ImGui::Separator();

          // Coefficient table
          if (ImGui::BeginTable(
                  "CoeffTable", 6,
                  ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                      ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit,
                  ImVec2(0, 320))) {

            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Idx", ImGuiTableColumnFlags_WidthFixed,
                                    40);
            ImGui::TableSetupColumn("[r,c]", ImGuiTableColumnFlags_WidthFixed,
                                    55);
            ImGui::TableSetupColumn("Original",
                                    ImGuiTableColumnFlags_WidthFixed, 90);
            ImGui::TableSetupColumn("Step", ImGuiTableColumnFlags_WidthFixed,
                                    70);
            ImGui::TableSetupColumn("Level", ImGuiTableColumnFlags_WidthFixed,
                                    55);
            ImGui::TableSetupColumn("Reconstructed",
                                    ImGuiTableColumnFlags_WidthFixed, 105);
            ImGui::TableHeadersRow();

            for (int i = 0; i < (int)res.entries.size(); i++) {
              auto &en = res.entries[i];
              ImGui::TableNextRow();

              // Highlight zeros differently
              if (en.level == 0)
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                       IM_COL32(40, 40, 40, 255));

              ImGui::TableSetColumnIndex(0);
              ImGui::Text("%d", i);
              ImGui::TableSetColumnIndex(1);
              ImGui::Text("[%d,%d]", i / N, i % N);
              ImGui::TableSetColumnIndex(2);
              // Color by magnitude
              float mag = fabsf(en.original);
              if (mag > 100)
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

      // ------------------------------------------------
      // TAB: Reconstructed — side-by-side view
      // ------------------------------------------------
      if (ImGui::BeginTabItem("Reconstructed")) {
        if (!g_experiment_run) {
          ImGui::TextDisabled("Run an experiment to see reconstructed output.");
        } else {
          float half = (ImGui::GetContentRegionAvail().x - 12) / 2.0f;
          float scale = half / (float)g_tex_w;
          float dw = g_tex_w * scale;
          float dh = g_tex_h * scale;

          ImGui::Text("Original");
          ImGui::SameLine(half + 12);
          ImGui::Text("Reconstructed (Q=%.0f, %s)", g_base_q,
                      (const char *[]){"Flat", "JPEG", "Ramp", "Deadzone",
                                       "Custom"}[(int)g_quant_mode]);

          ImGui::Image((ImTextureID)(intptr_t)g_tex_original, ImVec2(dw, dh));
          ImGui::SameLine();
          if (g_tex_recon)
            ImGui::Image((ImTextureID)(intptr_t)g_tex_recon, ImVec2(dw, dh));
        }
        ImGui::EndTabItem();
      }

      // ------------------------------------------------
      // TAB: Heatmaps
      // ------------------------------------------------
      if (ImGui::BeginTabItem("Heatmaps")) {
        if (g_dct_blocks.empty()) {
          ImGui::TextDisabled("Load an image first.");
        } else {
          ImGui::Text("Block energy heatmap (brighter = higher DCT energy)");
          float panel_w = ImGui::GetContentRegionAvail().x;
          float scale = panel_w / (float)g_tex_w;
          float bs = g_block_size * scale;

          ImVec2 base = ImGui::GetCursorScreenPos();
          ImGui::Dummy(ImVec2(g_tex_w * scale, g_tex_h * scale));
          ImDrawList *dl = ImGui::GetWindowDrawList();

          float max_e = 1.0f;
          for (auto &bl : g_dct_blocks)
            if (bl.energy > max_e)
              max_e = bl.energy;

          for (auto &bl : g_dct_blocks) {
            float px = base.x + bl.bx * bs;
            float py = base.y + bl.by * bs;
            float t = bl.energy / max_e;
            dl->AddRectFilled(ImVec2(px, py), ImVec2(px + bs, py + bs),
                              HeatColor(t, 230));
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

    ImGui::Render();
    glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
    glClearColor(bg.x, bg.y, bg.z, 1.0f);
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
