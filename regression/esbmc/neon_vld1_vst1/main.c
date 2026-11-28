// vld1/vst1 for u8, s16, u32, f32 and u64 in d and q widths.
// The asserted values are those the same code produces natively on an
// AArch64 host (clang -O0).
#include <arm_neon.h>
#include <stdint.h>

int main(void)
{
  uint8_t a8[16], b8[16], c8[16];
  int16_t s16[8];
  uint32_t u32[4];
  float f32[4];
  uint64_t u64[2];
  for (int i = 0; i < 16; i++)
  {
    a8[i] = i;
    b8[i] = 2 * i;
  }
  for (int i = 0; i < 8; i++)
    s16[i] = (int16_t)(-1000 * i + 7);
  for (int i = 0; i < 4; i++)
  {
    u32[i] = 0x01020304u * (i + 1);
    f32[i] = 1.5f * i - 2.25f;
  }
  u64[0] = 0x0123456789abcdefull;
  u64[1] = 0xfedcba9876543210ull;

  uint8x16_t va = vld1q_u8(a8), vb = vld1q_u8(b8);
  vst1q_u8(c8, vaddq_u8(va, vb));
  int16x8_t q16 = vld1q_s16(s16);
  int16x4_t d16 = vld1_s16(s16 + 4);
  uint32x4_t q32 = vld1q_u32(u32);
  float32x4_t qf = vld1q_f32(f32);
  float32x2_t df = vld1_f32(f32 + 2);
  uint64x2_t q64 = vld1q_u64(u64);
  uint64x1_t d64 = vld1_u64(u64 + 1);

  int16_t o16[12];
  uint32_t o32[4];
  float of[6];
  uint64_t o64[3];
  vst1q_s16(o16, q16);
  vst1_s16(o16 + 8, d16);
  vst1q_u32(o32, vaddq_u32(q32, q32));
  vst1q_f32(of, qf);
  vst1_f32(of + 4, df);
  vst1q_u64(o64, q64);
  vst1_u64(o64 + 2, d64);
  __ESBMC_assert(c8[0] == 0ull, "c8[0]");
  __ESBMC_assert(c8[1] == 3ull, "c8[1]");
  __ESBMC_assert(c8[2] == 6ull, "c8[2]");
  __ESBMC_assert(c8[3] == 9ull, "c8[3]");
  __ESBMC_assert(c8[4] == 12ull, "c8[4]");
  __ESBMC_assert(c8[5] == 15ull, "c8[5]");
  __ESBMC_assert(c8[6] == 18ull, "c8[6]");
  __ESBMC_assert(c8[7] == 21ull, "c8[7]");
  __ESBMC_assert(c8[8] == 24ull, "c8[8]");
  __ESBMC_assert(c8[9] == 27ull, "c8[9]");
  __ESBMC_assert(c8[10] == 30ull, "c8[10]");
  __ESBMC_assert(c8[11] == 33ull, "c8[11]");
  __ESBMC_assert(c8[12] == 36ull, "c8[12]");
  __ESBMC_assert(c8[13] == 39ull, "c8[13]");
  __ESBMC_assert(c8[14] == 42ull, "c8[14]");
  __ESBMC_assert(c8[15] == 45ull, "c8[15]");
  __ESBMC_assert(q16[0] == 7, "q16[0]");
  __ESBMC_assert(q16[1] == -993, "q16[1]");
  __ESBMC_assert(q16[2] == -1993, "q16[2]");
  __ESBMC_assert(q16[3] == -2993, "q16[3]");
  __ESBMC_assert(q16[4] == -3993, "q16[4]");
  __ESBMC_assert(q16[5] == -4993, "q16[5]");
  __ESBMC_assert(q16[6] == -5993, "q16[6]");
  __ESBMC_assert(q16[7] == -6993, "q16[7]");
  __ESBMC_assert(d16[0] == -3993, "d16[0]");
  __ESBMC_assert(d16[1] == -4993, "d16[1]");
  __ESBMC_assert(d16[2] == -5993, "d16[2]");
  __ESBMC_assert(d16[3] == -6993, "d16[3]");
  __ESBMC_assert(q32[0] == 16909060ull, "q32[0]");
  __ESBMC_assert(q32[1] == 33818120ull, "q32[1]");
  __ESBMC_assert(q32[2] == 50727180ull, "q32[2]");
  __ESBMC_assert(q32[3] == 67636240ull, "q32[3]");
  __ESBMC_assert(qf[0] == -0x1.2p+1f, "qf[0]");
  __ESBMC_assert(qf[1] == -0x1.8p-1f, "qf[1]");
  __ESBMC_assert(qf[2] == 0x1.8p-1f, "qf[2]");
  __ESBMC_assert(qf[3] == 0x1.2p+1f, "qf[3]");
  __ESBMC_assert(df[0] == 0x1.8p-1f, "df[0]");
  __ESBMC_assert(df[1] == 0x1.2p+1f, "df[1]");
  __ESBMC_assert(q64[0] == 81985529216486895ull, "q64[0]");
  __ESBMC_assert(q64[1] == 18364758544493064720ull, "q64[1]");
  __ESBMC_assert(d64[0] == 18364758544493064720ull, "d64[0]");
  __ESBMC_assert(o16[0] == 7, "o16[0]");
  __ESBMC_assert(o16[1] == -993, "o16[1]");
  __ESBMC_assert(o16[2] == -1993, "o16[2]");
  __ESBMC_assert(o16[3] == -2993, "o16[3]");
  __ESBMC_assert(o16[4] == -3993, "o16[4]");
  __ESBMC_assert(o16[5] == -4993, "o16[5]");
  __ESBMC_assert(o16[6] == -5993, "o16[6]");
  __ESBMC_assert(o16[7] == -6993, "o16[7]");
  __ESBMC_assert(o16[8] == -3993, "o16[8]");
  __ESBMC_assert(o16[9] == -4993, "o16[9]");
  __ESBMC_assert(o16[10] == -5993, "o16[10]");
  __ESBMC_assert(o16[11] == -6993, "o16[11]");
  __ESBMC_assert(o32[0] == 33818120ull, "o32[0]");
  __ESBMC_assert(o32[1] == 67636240ull, "o32[1]");
  __ESBMC_assert(o32[2] == 101454360ull, "o32[2]");
  __ESBMC_assert(o32[3] == 135272480ull, "o32[3]");
  __ESBMC_assert(of[0] == -0x1.2p+1f, "of[0]");
  __ESBMC_assert(of[1] == -0x1.8p-1f, "of[1]");
  __ESBMC_assert(of[2] == 0x1.8p-1f, "of[2]");
  __ESBMC_assert(of[3] == 0x1.2p+1f, "of[3]");
  __ESBMC_assert(of[4] == 0x1.8p-1f, "of[4]");
  __ESBMC_assert(of[5] == 0x1.2p+1f, "of[5]");
  __ESBMC_assert(o64[0] == 81985529216486895ull, "o64[0]");
  __ESBMC_assert(o64[1] == 18364758544493064720ull, "o64[1]");
  __ESBMC_assert(o64[2] == 18364758544493064720ull, "o64[2]");
  return 0;
}
