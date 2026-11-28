// arm_neon.h's loads and stores from C++, where each __builtin_bit_cast of a
// load's result has a materialised temporary as operand. The asserted values
// are those the same code produces natively on an AArch64 host.
#include <arm_neon.h>
#include <stdint.h>

template <typename T> static long lane_sum(const T *p, int n)
{
  long s = 0;
  for (int i = 0; i < n; i++)
    s += p[i];
  return s;
}

int main()
{
  uint8_t a[16], c[16];
  int16_t s[24], o[24];
  float f[8];
  for (int i = 0; i < 16; i++)
    a[i] = 9 * i;
  for (int i = 0; i < 24; i++)
    s[i] = (int16_t)(1000 - 97 * i);
  for (int i = 0; i < 8; i++)
    f[i] = 0.5f * i - 1.0f;

  uint8x16_t v = vld1q_u8(a);
  vst1q_u8(c, vaddq_u8(v, v));
  int16x8x3_t d = vld3q_s16(s);
  vst3q_s16(o, d);
  float32x4x2_t x = vld1q_f32_x2(f);

  __ESBMC_assert(c[15] == 14, "c[15]");
  __ESBMC_assert(lane_sum(c, 16) == 1904, "sum c");
  __ESBMC_assert(d.val[0][1] == 709, "d.val[0][1]");
  __ESBMC_assert(d.val[2][7] == -1231, "d.val[2][7]");
  __ESBMC_assert(lane_sum(o, 24) == lane_sum(s, 24), "vst3q_s16 round trip");
  __ESBMC_assert(o[23] == s[23], "o[23]");
  __ESBMC_assert(x.val[1][3] == 2.5f, "x.val[1][3]");
  return 0;
}
