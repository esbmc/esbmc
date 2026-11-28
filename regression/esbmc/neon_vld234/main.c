// vld2/vld3/vld4 de-interleave memory into registers by element size.
// The asserted values are those the same code produces natively on an
// AArch64 host (clang -O0).
#include <arm_neon.h>
#include <stdint.h>

int main(void)
{
  uint8_t a8[48];
  int16_t s16[24];
  float f32[16];
  uint32_t u32[4];
  uint64_t u64[8];
  for (int i = 0; i < 48; i++)
    a8[i] = 5 * i;
  for (int i = 0; i < 24; i++)
    s16[i] = (int16_t)(-30000 + 2500 * i);
  for (int i = 0; i < 16; i++)
    f32[i] = 0.5f * i + 0.125f;
  for (int i = 0; i < 4; i++)
    u32[i] = 0xdead0000u + i;
  for (int i = 0; i < 8; i++)
    u64[i] = 0x0100000000000001ull * i;

  uint8x16x2_t l2 = vld2q_u8(a8);
  int16x8x3_t l3 = vld3q_s16(s16);
  float32x4x4_t l4 = vld4q_f32(f32);
  uint32x2x2_t d2 = vld2_u32(u32);
  uint8x8x3_t d3 = vld3_u8(a8);
  uint64x2x4_t q4 = vld4q_u64(u64);
  __ESBMC_assert(l2.val[0][0] == 0ull, "l2.val[0][0]");
  __ESBMC_assert(l2.val[0][1] == 10ull, "l2.val[0][1]");
  __ESBMC_assert(l2.val[0][2] == 20ull, "l2.val[0][2]");
  __ESBMC_assert(l2.val[0][3] == 30ull, "l2.val[0][3]");
  __ESBMC_assert(l2.val[0][4] == 40ull, "l2.val[0][4]");
  __ESBMC_assert(l2.val[0][5] == 50ull, "l2.val[0][5]");
  __ESBMC_assert(l2.val[0][6] == 60ull, "l2.val[0][6]");
  __ESBMC_assert(l2.val[0][7] == 70ull, "l2.val[0][7]");
  __ESBMC_assert(l2.val[0][8] == 80ull, "l2.val[0][8]");
  __ESBMC_assert(l2.val[0][9] == 90ull, "l2.val[0][9]");
  __ESBMC_assert(l2.val[0][10] == 100ull, "l2.val[0][10]");
  __ESBMC_assert(l2.val[0][11] == 110ull, "l2.val[0][11]");
  __ESBMC_assert(l2.val[0][12] == 120ull, "l2.val[0][12]");
  __ESBMC_assert(l2.val[0][13] == 130ull, "l2.val[0][13]");
  __ESBMC_assert(l2.val[0][14] == 140ull, "l2.val[0][14]");
  __ESBMC_assert(l2.val[0][15] == 150ull, "l2.val[0][15]");
  __ESBMC_assert(l2.val[1][0] == 5ull, "l2.val[1][0]");
  __ESBMC_assert(l2.val[1][1] == 15ull, "l2.val[1][1]");
  __ESBMC_assert(l2.val[1][2] == 25ull, "l2.val[1][2]");
  __ESBMC_assert(l2.val[1][3] == 35ull, "l2.val[1][3]");
  __ESBMC_assert(l2.val[1][4] == 45ull, "l2.val[1][4]");
  __ESBMC_assert(l2.val[1][5] == 55ull, "l2.val[1][5]");
  __ESBMC_assert(l2.val[1][6] == 65ull, "l2.val[1][6]");
  __ESBMC_assert(l2.val[1][7] == 75ull, "l2.val[1][7]");
  __ESBMC_assert(l2.val[1][8] == 85ull, "l2.val[1][8]");
  __ESBMC_assert(l2.val[1][9] == 95ull, "l2.val[1][9]");
  __ESBMC_assert(l2.val[1][10] == 105ull, "l2.val[1][10]");
  __ESBMC_assert(l2.val[1][11] == 115ull, "l2.val[1][11]");
  __ESBMC_assert(l2.val[1][12] == 125ull, "l2.val[1][12]");
  __ESBMC_assert(l2.val[1][13] == 135ull, "l2.val[1][13]");
  __ESBMC_assert(l2.val[1][14] == 145ull, "l2.val[1][14]");
  __ESBMC_assert(l2.val[1][15] == 155ull, "l2.val[1][15]");
  __ESBMC_assert(l3.val[0][0] == -30000, "l3.val[0][0]");
  __ESBMC_assert(l3.val[0][1] == -22500, "l3.val[0][1]");
  __ESBMC_assert(l3.val[0][2] == -15000, "l3.val[0][2]");
  __ESBMC_assert(l3.val[0][3] == -7500, "l3.val[0][3]");
  __ESBMC_assert(l3.val[0][4] == 0, "l3.val[0][4]");
  __ESBMC_assert(l3.val[0][5] == 7500, "l3.val[0][5]");
  __ESBMC_assert(l3.val[0][6] == 15000, "l3.val[0][6]");
  __ESBMC_assert(l3.val[0][7] == 22500, "l3.val[0][7]");
  __ESBMC_assert(l3.val[1][0] == -27500, "l3.val[1][0]");
  __ESBMC_assert(l3.val[1][1] == -20000, "l3.val[1][1]");
  __ESBMC_assert(l3.val[1][2] == -12500, "l3.val[1][2]");
  __ESBMC_assert(l3.val[1][3] == -5000, "l3.val[1][3]");
  __ESBMC_assert(l3.val[1][4] == 2500, "l3.val[1][4]");
  __ESBMC_assert(l3.val[1][5] == 10000, "l3.val[1][5]");
  __ESBMC_assert(l3.val[1][6] == 17500, "l3.val[1][6]");
  __ESBMC_assert(l3.val[1][7] == 25000, "l3.val[1][7]");
  __ESBMC_assert(l3.val[2][0] == -25000, "l3.val[2][0]");
  __ESBMC_assert(l3.val[2][1] == -17500, "l3.val[2][1]");
  __ESBMC_assert(l3.val[2][2] == -10000, "l3.val[2][2]");
  __ESBMC_assert(l3.val[2][3] == -2500, "l3.val[2][3]");
  __ESBMC_assert(l3.val[2][4] == 5000, "l3.val[2][4]");
  __ESBMC_assert(l3.val[2][5] == 12500, "l3.val[2][5]");
  __ESBMC_assert(l3.val[2][6] == 20000, "l3.val[2][6]");
  __ESBMC_assert(l3.val[2][7] == 27500, "l3.val[2][7]");
  __ESBMC_assert(l4.val[0][0] == 0x1p-3f, "l4.val[0][0]");
  __ESBMC_assert(l4.val[0][1] == 0x1.1p+1f, "l4.val[0][1]");
  __ESBMC_assert(l4.val[0][2] == 0x1.08p+2f, "l4.val[0][2]");
  __ESBMC_assert(l4.val[0][3] == 0x1.88p+2f, "l4.val[0][3]");
  __ESBMC_assert(l4.val[1][0] == 0x1.4p-1f, "l4.val[1][0]");
  __ESBMC_assert(l4.val[1][1] == 0x1.5p+1f, "l4.val[1][1]");
  __ESBMC_assert(l4.val[1][2] == 0x1.28p+2f, "l4.val[1][2]");
  __ESBMC_assert(l4.val[1][3] == 0x1.a8p+2f, "l4.val[1][3]");
  __ESBMC_assert(l4.val[2][0] == 0x1.2p+0f, "l4.val[2][0]");
  __ESBMC_assert(l4.val[2][1] == 0x1.9p+1f, "l4.val[2][1]");
  __ESBMC_assert(l4.val[2][2] == 0x1.48p+2f, "l4.val[2][2]");
  __ESBMC_assert(l4.val[2][3] == 0x1.c8p+2f, "l4.val[2][3]");
  __ESBMC_assert(l4.val[3][0] == 0x1.ap+0f, "l4.val[3][0]");
  __ESBMC_assert(l4.val[3][1] == 0x1.dp+1f, "l4.val[3][1]");
  __ESBMC_assert(l4.val[3][2] == 0x1.68p+2f, "l4.val[3][2]");
  __ESBMC_assert(l4.val[3][3] == 0x1.e8p+2f, "l4.val[3][3]");
  __ESBMC_assert(d2.val[0][0] == 3735879680ull, "d2.val[0][0]");
  __ESBMC_assert(d2.val[0][1] == 3735879682ull, "d2.val[0][1]");
  __ESBMC_assert(d2.val[1][0] == 3735879681ull, "d2.val[1][0]");
  __ESBMC_assert(d2.val[1][1] == 3735879683ull, "d2.val[1][1]");
  __ESBMC_assert(d3.val[0][0] == 0ull, "d3.val[0][0]");
  __ESBMC_assert(d3.val[0][1] == 15ull, "d3.val[0][1]");
  __ESBMC_assert(d3.val[0][2] == 30ull, "d3.val[0][2]");
  __ESBMC_assert(d3.val[0][3] == 45ull, "d3.val[0][3]");
  __ESBMC_assert(d3.val[0][4] == 60ull, "d3.val[0][4]");
  __ESBMC_assert(d3.val[0][5] == 75ull, "d3.val[0][5]");
  __ESBMC_assert(d3.val[0][6] == 90ull, "d3.val[0][6]");
  __ESBMC_assert(d3.val[0][7] == 105ull, "d3.val[0][7]");
  __ESBMC_assert(d3.val[1][0] == 5ull, "d3.val[1][0]");
  __ESBMC_assert(d3.val[1][1] == 20ull, "d3.val[1][1]");
  __ESBMC_assert(d3.val[1][2] == 35ull, "d3.val[1][2]");
  __ESBMC_assert(d3.val[1][3] == 50ull, "d3.val[1][3]");
  __ESBMC_assert(d3.val[1][4] == 65ull, "d3.val[1][4]");
  __ESBMC_assert(d3.val[1][5] == 80ull, "d3.val[1][5]");
  __ESBMC_assert(d3.val[1][6] == 95ull, "d3.val[1][6]");
  __ESBMC_assert(d3.val[1][7] == 110ull, "d3.val[1][7]");
  __ESBMC_assert(d3.val[2][0] == 10ull, "d3.val[2][0]");
  __ESBMC_assert(d3.val[2][1] == 25ull, "d3.val[2][1]");
  __ESBMC_assert(d3.val[2][2] == 40ull, "d3.val[2][2]");
  __ESBMC_assert(d3.val[2][3] == 55ull, "d3.val[2][3]");
  __ESBMC_assert(d3.val[2][4] == 70ull, "d3.val[2][4]");
  __ESBMC_assert(d3.val[2][5] == 85ull, "d3.val[2][5]");
  __ESBMC_assert(d3.val[2][6] == 100ull, "d3.val[2][6]");
  __ESBMC_assert(d3.val[2][7] == 115ull, "d3.val[2][7]");
  __ESBMC_assert(q4.val[0][0] == 0ull, "q4.val[0][0]");
  __ESBMC_assert(q4.val[0][1] == 288230376151711748ull, "q4.val[0][1]");
  __ESBMC_assert(q4.val[1][0] == 72057594037927937ull, "q4.val[1][0]");
  __ESBMC_assert(q4.val[1][1] == 360287970189639685ull, "q4.val[1][1]");
  __ESBMC_assert(q4.val[2][0] == 144115188075855874ull, "q4.val[2][0]");
  __ESBMC_assert(q4.val[2][1] == 432345564227567622ull, "q4.val[2][1]");
  __ESBMC_assert(q4.val[3][0] == 216172782113783811ull, "q4.val[3][0]");
  __ESBMC_assert(q4.val[3][1] == 504403158265495559ull, "q4.val[3][1]");
  return 0;
}
