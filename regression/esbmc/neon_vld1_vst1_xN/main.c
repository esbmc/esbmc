// vld1_xN/vst1_xN: two to four registers, contiguous in memory.
// The asserted values are those the same code produces natively on an
// AArch64 host (clang -O0).
#include <arm_neon.h>
#include <stdint.h>

int main(void)
{
  uint8_t a8[32];
  int16_t s16[12];
  float f32[16];
  uint32_t u32[8];
  uint64_t u64[3];
  for (int i = 0; i < 32; i++)
    a8[i] = 3 * i + 1;
  for (int i = 0; i < 12; i++)
    s16[i] = (int16_t)(100 - 37 * i);
  for (int i = 0; i < 16; i++)
    f32[i] = 0.25f * i - 1.0f;
  for (int i = 0; i < 8; i++)
    u32[i] = 0x10000001u * i;
  for (int i = 0; i < 3; i++)
    u64[i] = 0x1111111111111111ull * (i + 1);

  uint8x16x2_t x2 = vld1q_u8_x2(a8);
  int16x4x3_t x3 = vld1_s16_x3(s16);
  float32x4x4_t x4 = vld1q_f32_x4(f32);

  uint32_t o32[8];
  uint64_t o64[3];
  float of[16];
  vst1q_u32_x2(o32, vld1q_u32_x2(u32));
  vst1_u64_x3(o64, vld1_u64_x3(u64));
  vst1q_f32_x4(of, x4);
  __ESBMC_assert(x2.val[0][0] == 1ull, "x2.val[0][0]");
  __ESBMC_assert(x2.val[0][1] == 4ull, "x2.val[0][1]");
  __ESBMC_assert(x2.val[0][2] == 7ull, "x2.val[0][2]");
  __ESBMC_assert(x2.val[0][3] == 10ull, "x2.val[0][3]");
  __ESBMC_assert(x2.val[0][4] == 13ull, "x2.val[0][4]");
  __ESBMC_assert(x2.val[0][5] == 16ull, "x2.val[0][5]");
  __ESBMC_assert(x2.val[0][6] == 19ull, "x2.val[0][6]");
  __ESBMC_assert(x2.val[0][7] == 22ull, "x2.val[0][7]");
  __ESBMC_assert(x2.val[0][8] == 25ull, "x2.val[0][8]");
  __ESBMC_assert(x2.val[0][9] == 28ull, "x2.val[0][9]");
  __ESBMC_assert(x2.val[0][10] == 31ull, "x2.val[0][10]");
  __ESBMC_assert(x2.val[0][11] == 34ull, "x2.val[0][11]");
  __ESBMC_assert(x2.val[0][12] == 37ull, "x2.val[0][12]");
  __ESBMC_assert(x2.val[0][13] == 40ull, "x2.val[0][13]");
  __ESBMC_assert(x2.val[0][14] == 43ull, "x2.val[0][14]");
  __ESBMC_assert(x2.val[0][15] == 46ull, "x2.val[0][15]");
  __ESBMC_assert(x2.val[1][0] == 49ull, "x2.val[1][0]");
  __ESBMC_assert(x2.val[1][1] == 52ull, "x2.val[1][1]");
  __ESBMC_assert(x2.val[1][2] == 55ull, "x2.val[1][2]");
  __ESBMC_assert(x2.val[1][3] == 58ull, "x2.val[1][3]");
  __ESBMC_assert(x2.val[1][4] == 61ull, "x2.val[1][4]");
  __ESBMC_assert(x2.val[1][5] == 64ull, "x2.val[1][5]");
  __ESBMC_assert(x2.val[1][6] == 67ull, "x2.val[1][6]");
  __ESBMC_assert(x2.val[1][7] == 70ull, "x2.val[1][7]");
  __ESBMC_assert(x2.val[1][8] == 73ull, "x2.val[1][8]");
  __ESBMC_assert(x2.val[1][9] == 76ull, "x2.val[1][9]");
  __ESBMC_assert(x2.val[1][10] == 79ull, "x2.val[1][10]");
  __ESBMC_assert(x2.val[1][11] == 82ull, "x2.val[1][11]");
  __ESBMC_assert(x2.val[1][12] == 85ull, "x2.val[1][12]");
  __ESBMC_assert(x2.val[1][13] == 88ull, "x2.val[1][13]");
  __ESBMC_assert(x2.val[1][14] == 91ull, "x2.val[1][14]");
  __ESBMC_assert(x2.val[1][15] == 94ull, "x2.val[1][15]");
  __ESBMC_assert(x3.val[0][0] == 100, "x3.val[0][0]");
  __ESBMC_assert(x3.val[0][1] == 63, "x3.val[0][1]");
  __ESBMC_assert(x3.val[0][2] == 26, "x3.val[0][2]");
  __ESBMC_assert(x3.val[0][3] == -11, "x3.val[0][3]");
  __ESBMC_assert(x3.val[1][0] == -48, "x3.val[1][0]");
  __ESBMC_assert(x3.val[1][1] == -85, "x3.val[1][1]");
  __ESBMC_assert(x3.val[1][2] == -122, "x3.val[1][2]");
  __ESBMC_assert(x3.val[1][3] == -159, "x3.val[1][3]");
  __ESBMC_assert(x3.val[2][0] == -196, "x3.val[2][0]");
  __ESBMC_assert(x3.val[2][1] == -233, "x3.val[2][1]");
  __ESBMC_assert(x3.val[2][2] == -270, "x3.val[2][2]");
  __ESBMC_assert(x3.val[2][3] == -307, "x3.val[2][3]");
  __ESBMC_assert(x4.val[0][0] == -0x1p+0f, "x4.val[0][0]");
  __ESBMC_assert(x4.val[0][1] == -0x1.8p-1f, "x4.val[0][1]");
  __ESBMC_assert(x4.val[0][2] == -0x1p-1f, "x4.val[0][2]");
  __ESBMC_assert(x4.val[0][3] == -0x1p-2f, "x4.val[0][3]");
  __ESBMC_assert(x4.val[1][0] == 0x0p+0f, "x4.val[1][0]");
  __ESBMC_assert(x4.val[1][1] == 0x1p-2f, "x4.val[1][1]");
  __ESBMC_assert(x4.val[1][2] == 0x1p-1f, "x4.val[1][2]");
  __ESBMC_assert(x4.val[1][3] == 0x1.8p-1f, "x4.val[1][3]");
  __ESBMC_assert(x4.val[2][0] == 0x1p+0f, "x4.val[2][0]");
  __ESBMC_assert(x4.val[2][1] == 0x1.4p+0f, "x4.val[2][1]");
  __ESBMC_assert(x4.val[2][2] == 0x1.8p+0f, "x4.val[2][2]");
  __ESBMC_assert(x4.val[2][3] == 0x1.cp+0f, "x4.val[2][3]");
  __ESBMC_assert(x4.val[3][0] == 0x1p+1f, "x4.val[3][0]");
  __ESBMC_assert(x4.val[3][1] == 0x1.2p+1f, "x4.val[3][1]");
  __ESBMC_assert(x4.val[3][2] == 0x1.4p+1f, "x4.val[3][2]");
  __ESBMC_assert(x4.val[3][3] == 0x1.6p+1f, "x4.val[3][3]");
  __ESBMC_assert(o32[0] == 0ull, "o32[0]");
  __ESBMC_assert(o32[1] == 268435457ull, "o32[1]");
  __ESBMC_assert(o32[2] == 536870914ull, "o32[2]");
  __ESBMC_assert(o32[3] == 805306371ull, "o32[3]");
  __ESBMC_assert(o32[4] == 1073741828ull, "o32[4]");
  __ESBMC_assert(o32[5] == 1342177285ull, "o32[5]");
  __ESBMC_assert(o32[6] == 1610612742ull, "o32[6]");
  __ESBMC_assert(o32[7] == 1879048199ull, "o32[7]");
  __ESBMC_assert(o64[0] == 1229782938247303441ull, "o64[0]");
  __ESBMC_assert(o64[1] == 2459565876494606882ull, "o64[1]");
  __ESBMC_assert(o64[2] == 3689348814741910323ull, "o64[2]");
  __ESBMC_assert(of[0] == -0x1p+0f, "of[0]");
  __ESBMC_assert(of[1] == -0x1.8p-1f, "of[1]");
  __ESBMC_assert(of[2] == -0x1p-1f, "of[2]");
  __ESBMC_assert(of[3] == -0x1p-2f, "of[3]");
  __ESBMC_assert(of[4] == 0x0p+0f, "of[4]");
  __ESBMC_assert(of[5] == 0x1p-2f, "of[5]");
  __ESBMC_assert(of[6] == 0x1p-1f, "of[6]");
  __ESBMC_assert(of[7] == 0x1.8p-1f, "of[7]");
  __ESBMC_assert(of[8] == 0x1p+0f, "of[8]");
  __ESBMC_assert(of[9] == 0x1.4p+0f, "of[9]");
  __ESBMC_assert(of[10] == 0x1.8p+0f, "of[10]");
  __ESBMC_assert(of[11] == 0x1.cp+0f, "of[11]");
  __ESBMC_assert(of[12] == 0x1p+1f, "of[12]");
  __ESBMC_assert(of[13] == 0x1.2p+1f, "of[13]");
  __ESBMC_assert(of[14] == 0x1.4p+1f, "of[14]");
  __ESBMC_assert(of[15] == 0x1.6p+1f, "of[15]");
  return 0;
}
