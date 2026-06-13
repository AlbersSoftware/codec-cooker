#include "dct.h"
#include <cmath>

#define PI 3.14159265358979323846

static float C(int u) { return (u == 0) ? (1.0f / sqrtf(2.0f)) : 1.0f; }

// RGBA → grayscale
std::vector<unsigned char> ExtractGrayscale(const unsigned char *rgba,
                                            int width, int height) {
  std::vector<unsigned char> gray(width * height);

  for (int i = 0; i < width * height; i++) {
    int r = rgba[i * 4 + 0];
    int g = rgba[i * 4 + 1];
    int b = rgba[i * 4 + 2];

    gray[i] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
  }

  return gray;
}

std::vector<DCTBlock> ComputeDCTBlocks(const std::vector<unsigned char> &gray,
                                       int width, int height, int blockSize) {
  std::vector<DCTBlock> blocks;

  for (int by = 0; by < height; by += blockSize) {
    for (int bx = 0; bx < width; bx += blockSize) {

      DCTBlock block;
      block.bx = bx / blockSize;
      block.by = by / blockSize;
      block.coeffs.resize(blockSize * blockSize);

      float energy = 0.0f;

      for (int u = 0; u < blockSize; u++) {
        for (int v = 0; v < blockSize; v++) {

          float sum = 0.0f;

          for (int x = 0; x < blockSize; x++) {
            for (int y = 0; y < blockSize; y++) {

              int ix = bx + x;
              int iy = by + y;

              if (ix >= width || iy >= height)
                continue;

              float pixel = gray[iy * width + ix] - 128.0f;

              sum += pixel * cosf((2 * x + 1) * u * PI / (2 * blockSize)) *
                     cosf((2 * y + 1) * v * PI / (2 * blockSize));
            }
          }

          float cu = C(u);
          float cv = C(v);

          float coeff = 0.25f * cu * cv * sum;

          block.coeffs[v * blockSize + u] = coeff;
          energy += coeff * coeff;
        }
      }

      block.energy = energy;
      blocks.push_back(block);
    }
  }

  return blocks;
}
