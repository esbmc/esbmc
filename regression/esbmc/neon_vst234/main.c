// vst2/vst3/vst4 interleave registers into memory by element size.
// The asserted values are those the same code produces natively on an
// AArch64 host (clang -O0).
#include <arm_neon.h>
#include <stdint.h>

int main(void)
{
  uint8_t a8[32];
  int16_t s16[12];
  float f32[16];
  uint64_t u64[6];
  for (int i = 0; i < 32; i++)
    a8[i] = 7 * i + 3;
  for (int i = 0; i < 12; i++)
    s16[i] = (int16_t)(-6 * i * i + 11);
  for (int i = 0; i < 16; i++)
    f32[i] = -0.75f * i + 4.0f;
  for (int i = 0; i < 6; i++)
    u64[i] = 0xa5a5a5a500000000ull + i;

  uint8x16x2_t r2 = vld1q_u8_x2(a8);
  int16x4x3_t r3 = vld1_s16_x3(s16);
  float32x4x4_t r4 = vld1q_f32_x4(f32);
  uint64x2x3_t r6 = vld1q_u64_x3(u64);

  uint8_t o8[32];
  int16_t o16[12];
  float of[16];
  uint64_t o64[6];
  vst2q_u8(o8, r2);
  vst3_s16(o16, r3);
  vst4q_f32(of, r4);
  vst3q_u64(o64, r6);
  __ESBMC_assert(o8[0] == 3ull, "o8[0]");
  __ESBMC_assert(o8[1] == 115ull, "o8[1]");
  __ESBMC_assert(o8[2] == 10ull, "o8[2]");
  __ESBMC_assert(o8[3] == 122ull, "o8[3]");
  __ESBMC_assert(o8[4] == 17ull, "o8[4]");
  __ESBMC_assert(o8[5] == 129ull, "o8[5]");
  __ESBMC_assert(o8[6] == 24ull, "o8[6]");
  __ESBMC_assert(o8[7] == 136ull, "o8[7]");
  __ESBMC_assert(o8[8] == 31ull, "o8[8]");
  __ESBMC_assert(o8[9] == 143ull, "o8[9]");
  __ESBMC_assert(o8[10] == 38ull, "o8[10]");
  __ESBMC_assert(o8[11] == 150ull, "o8[11]");
  __ESBMC_assert(o8[12] == 45ull, "o8[12]");
  __ESBMC_assert(o8[13] == 157ull, "o8[13]");
  __ESBMC_assert(o8[14] == 52ull, "o8[14]");
  __ESBMC_assert(o8[15] == 164ull, "o8[15]");
  __ESBMC_assert(o8[16] == 59ull, "o8[16]");
  __ESBMC_assert(o8[17] == 171ull, "o8[17]");
  __ESBMC_assert(o8[18] == 66ull, "o8[18]");
  __ESBMC_assert(o8[19] == 178ull, "o8[19]");
  __ESBMC_assert(o8[20] == 73ull, "o8[20]");
  __ESBMC_assert(o8[21] == 185ull, "o8[21]");
  __ESBMC_assert(o8[22] == 80ull, "o8[22]");
  __ESBMC_assert(o8[23] == 192ull, "o8[23]");
  __ESBMC_assert(o8[24] == 87ull, "o8[24]");
  __ESBMC_assert(o8[25] == 199ull, "o8[25]");
  __ESBMC_assert(o8[26] == 94ull, "o8[26]");
  __ESBMC_assert(o8[27] == 206ull, "o8[27]");
  __ESBMC_assert(o8[28] == 101ull, "o8[28]");
  __ESBMC_assert(o8[29] == 213ull, "o8[29]");
  __ESBMC_assert(o8[30] == 108ull, "o8[30]");
  __ESBMC_assert(o8[31] == 220ull, "o8[31]");
  __ESBMC_assert(o16[0] == 11, "o16[0]");
  __ESBMC_assert(o16[1] == -85, "o16[1]");
  __ESBMC_assert(o16[2] == -373, "o16[2]");
  __ESBMC_assert(o16[3] == 5, "o16[3]");
  __ESBMC_assert(o16[4] == -139, "o16[4]");
  __ESBMC_assert(o16[5] == -475, "o16[5]");
  __ESBMC_assert(o16[6] == -13, "o16[6]");
  __ESBMC_assert(o16[7] == -205, "o16[7]");
  __ESBMC_assert(o16[8] == -589, "o16[8]");
  __ESBMC_assert(o16[9] == -43, "o16[9]");
  __ESBMC_assert(o16[10] == -283, "o16[10]");
  __ESBMC_assert(o16[11] == -715, "o16[11]");
  __ESBMC_assert(of[0] == 0x1p+2f, "of[0]");
  __ESBMC_assert(of[1] == 0x1p+0f, "of[1]");
  __ESBMC_assert(of[2] == -0x1p+1f, "of[2]");
  __ESBMC_assert(of[3] == -0x1.4p+2f, "of[3]");
  __ESBMC_assert(of[4] == 0x1.ap+1f, "of[4]");
  __ESBMC_assert(of[5] == 0x1p-2f, "of[5]");
  __ESBMC_assert(of[6] == -0x1.6p+1f, "of[6]");
  __ESBMC_assert(of[7] == -0x1.7p+2f, "of[7]");
  __ESBMC_assert(of[8] == 0x1.4p+1f, "of[8]");
  __ESBMC_assert(of[9] == -0x1p-1f, "of[9]");
  __ESBMC_assert(of[10] == -0x1.cp+1f, "of[10]");
  __ESBMC_assert(of[11] == -0x1.ap+2f, "of[11]");
  __ESBMC_assert(of[12] == 0x1.cp+0f, "of[12]");
  __ESBMC_assert(of[13] == -0x1.4p+0f, "of[13]");
  __ESBMC_assert(of[14] == -0x1.1p+2f, "of[14]");
  __ESBMC_assert(of[15] == -0x1.dp+2f, "of[15]");
  __ESBMC_assert(o64[0] == 11936128515503554560ull, "o64[0]");
  __ESBMC_assert(o64[1] == 11936128515503554562ull, "o64[1]");
  __ESBMC_assert(o64[2] == 11936128515503554564ull, "o64[2]");
  __ESBMC_assert(o64[3] == 11936128515503554561ull, "o64[3]");
  __ESBMC_assert(o64[4] == 11936128515503554563ull, "o64[4]");
  __ESBMC_assert(o64[5] == 11936128515503554565ull, "o64[5]");
  return 0;
}
