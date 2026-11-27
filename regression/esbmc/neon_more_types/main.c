// Loads and stores for s8, p8, p16, f16, s64 and f64 lanes.
// The asserted values are those the same code produces natively on an
// AArch64 host (clang -O0).
#include <arm_neon.h>
#include <stdint.h>

int main(void)
{
  int8_t s8[16];
  poly16_t p16[8];
  float16_t h[8];
  double f64[4];
  int64_t s64[4];
  poly8_t p8[24];
  for (int i = 0; i < 16; i++)
    s8[i] = (int8_t)(-100 + 13 * i);
  for (int i = 0; i < 8; i++)
  {
    p16[i] = 0x8000 + 77 * i;
    h[i] = (float16_t)(0.5f * i - 1.25f);
  }
  for (int i = 0; i < 4; i++)
  {
    f64[i] = -3.0 / (i + 1);
    s64[i] = -(1ll << 40) * (i + 1) + 3;
  }
  for (int i = 0; i < 24; i++)
    p8[i] = 11 * i;

  int8x16_t vs8 = vld1q_s8(s8);
  poly16x4x2_t vp16 = vld2_p16(p16);
  float64x2x2_t vf64 = vld2q_f64(f64);
  int64x1x4_t vs64 = vld4_s64(s64);
  float16x8_t vh = vld1q_f16(h);
  poly8x8x3_t vp8 = vld3_p8(p8);

  float16_t oh[8];
  double of[4];
  poly8_t op8[24];
  vst1q_f16(oh, vh);
  vst2q_f64(of, vf64);
  vst3_p8(op8, vp8);
  __ESBMC_assert(vs8[0] == -100, "vs8[0]");
  __ESBMC_assert(vs8[1] == -87, "vs8[1]");
  __ESBMC_assert(vs8[2] == -74, "vs8[2]");
  __ESBMC_assert(vs8[3] == -61, "vs8[3]");
  __ESBMC_assert(vs8[4] == -48, "vs8[4]");
  __ESBMC_assert(vs8[5] == -35, "vs8[5]");
  __ESBMC_assert(vs8[6] == -22, "vs8[6]");
  __ESBMC_assert(vs8[7] == -9, "vs8[7]");
  __ESBMC_assert(vs8[8] == 4, "vs8[8]");
  __ESBMC_assert(vs8[9] == 17, "vs8[9]");
  __ESBMC_assert(vs8[10] == 30, "vs8[10]");
  __ESBMC_assert(vs8[11] == 43, "vs8[11]");
  __ESBMC_assert(vs8[12] == 56, "vs8[12]");
  __ESBMC_assert(vs8[13] == 69, "vs8[13]");
  __ESBMC_assert(vs8[14] == 82, "vs8[14]");
  __ESBMC_assert(vs8[15] == 95, "vs8[15]");
  __ESBMC_assert(vp16.val[0][0] == 32768ull, "vp16.val[0][0]");
  __ESBMC_assert(vp16.val[0][1] == 32922ull, "vp16.val[0][1]");
  __ESBMC_assert(vp16.val[0][2] == 33076ull, "vp16.val[0][2]");
  __ESBMC_assert(vp16.val[0][3] == 33230ull, "vp16.val[0][3]");
  __ESBMC_assert(vp16.val[1][0] == 32845ull, "vp16.val[1][0]");
  __ESBMC_assert(vp16.val[1][1] == 32999ull, "vp16.val[1][1]");
  __ESBMC_assert(vp16.val[1][2] == 33153ull, "vp16.val[1][2]");
  __ESBMC_assert(vp16.val[1][3] == 33307ull, "vp16.val[1][3]");
  __ESBMC_assert(vf64.val[0][0] == -0x1.8p+1, "vf64.val[0][0]");
  __ESBMC_assert(vf64.val[0][1] == -0x1p+0, "vf64.val[0][1]");
  __ESBMC_assert(vf64.val[1][0] == -0x1.8p+0, "vf64.val[1][0]");
  __ESBMC_assert(vf64.val[1][1] == -0x1.8p-1, "vf64.val[1][1]");
  __ESBMC_assert(vs64.val[0][0] == -1099511627773, "vs64.val[0][0]");
  __ESBMC_assert(vs64.val[1][0] == -2199023255549, "vs64.val[1][0]");
  __ESBMC_assert(vs64.val[2][0] == -3298534883325, "vs64.val[2][0]");
  __ESBMC_assert(vs64.val[3][0] == -4398046511101, "vs64.val[3][0]");
  __ESBMC_assert((float)vh[0] == -0x1.4p+0f, "(float)vh[0]");
  __ESBMC_assert((float)vh[1] == -0x1.8p-1f, "(float)vh[1]");
  __ESBMC_assert((float)vh[2] == -0x1p-2f, "(float)vh[2]");
  __ESBMC_assert((float)vh[3] == 0x1p-2f, "(float)vh[3]");
  __ESBMC_assert((float)vh[4] == 0x1.8p-1f, "(float)vh[4]");
  __ESBMC_assert((float)vh[5] == 0x1.4p+0f, "(float)vh[5]");
  __ESBMC_assert((float)vh[6] == 0x1.cp+0f, "(float)vh[6]");
  __ESBMC_assert((float)vh[7] == 0x1.2p+1f, "(float)vh[7]");
  __ESBMC_assert((float)oh[0] == -0x1.4p+0f, "(float)oh[0]");
  __ESBMC_assert((float)oh[1] == -0x1.8p-1f, "(float)oh[1]");
  __ESBMC_assert((float)oh[2] == -0x1p-2f, "(float)oh[2]");
  __ESBMC_assert((float)oh[3] == 0x1p-2f, "(float)oh[3]");
  __ESBMC_assert((float)oh[4] == 0x1.8p-1f, "(float)oh[4]");
  __ESBMC_assert((float)oh[5] == 0x1.4p+0f, "(float)oh[5]");
  __ESBMC_assert((float)oh[6] == 0x1.cp+0f, "(float)oh[6]");
  __ESBMC_assert((float)oh[7] == 0x1.2p+1f, "(float)oh[7]");
  __ESBMC_assert(of[0] == -0x1.8p+1, "of[0]");
  __ESBMC_assert(of[1] == -0x1.8p+0, "of[1]");
  __ESBMC_assert(of[2] == -0x1p+0, "of[2]");
  __ESBMC_assert(of[3] == -0x1.8p-1, "of[3]");
  __ESBMC_assert(op8[0] == 0ull, "op8[0]");
  __ESBMC_assert(op8[1] == 11ull, "op8[1]");
  __ESBMC_assert(op8[2] == 22ull, "op8[2]");
  __ESBMC_assert(op8[3] == 33ull, "op8[3]");
  __ESBMC_assert(op8[4] == 44ull, "op8[4]");
  __ESBMC_assert(op8[5] == 55ull, "op8[5]");
  __ESBMC_assert(op8[6] == 66ull, "op8[6]");
  __ESBMC_assert(op8[7] == 77ull, "op8[7]");
  __ESBMC_assert(op8[8] == 88ull, "op8[8]");
  __ESBMC_assert(op8[9] == 99ull, "op8[9]");
  __ESBMC_assert(op8[10] == 110ull, "op8[10]");
  __ESBMC_assert(op8[11] == 121ull, "op8[11]");
  __ESBMC_assert(op8[12] == 132ull, "op8[12]");
  __ESBMC_assert(op8[13] == 143ull, "op8[13]");
  __ESBMC_assert(op8[14] == 154ull, "op8[14]");
  __ESBMC_assert(op8[15] == 165ull, "op8[15]");
  __ESBMC_assert(op8[16] == 176ull, "op8[16]");
  __ESBMC_assert(op8[17] == 187ull, "op8[17]");
  __ESBMC_assert(op8[18] == 198ull, "op8[18]");
  __ESBMC_assert(op8[19] == 209ull, "op8[19]");
  __ESBMC_assert(op8[20] == 220ull, "op8[20]");
  __ESBMC_assert(op8[21] == 231ull, "op8[21]");
  __ESBMC_assert(op8[22] == 242ull, "op8[22]");
  __ESBMC_assert(op8[23] == 253ull, "op8[23]");
  return 0;
}
