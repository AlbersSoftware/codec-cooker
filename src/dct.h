#pragma once
#include <vector>

struct DCTBlock {
  int bx;
  int by;
  float energy;
  std::vector<float> coeffs; // 8x8 = 64
};

std::vector<unsigned char> ExtractGrayscale(const unsigned char *rgba,
                                            int width, int height);

std::vector<DCTBlock> ComputeDCTBlocks(const std::vector<unsigned char> &gray,
                                       int width, int height, int blockSize);
