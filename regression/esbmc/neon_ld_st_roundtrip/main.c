// An interleaved load followed by the matching store, and a contiguous
// multi-register load and store, give back arbitrary memory unchanged.
#include <arm_neon.h>
#include <stdint.h>

int main(void)
{
  uint16_t a[24], b[24];
  uint8_t c[64], d[64];
  int i = nondet_int();
  __ESBMC_assume(0 <= i && i < 24);
  int j = nondet_int();
  __ESBMC_assume(0 <= j && j < 64);
  vst3q_u16(b, vld3q_u16(a));
  vst4q_u8(d, vld4q_u8(c));
  __ESBMC_assert(a[i] == b[i], "vld3q_u16/vst3q_u16 round trip");
  __ESBMC_assert(c[j] == d[j], "vld4q_u8/vst4q_u8 round trip");
  return 0;
}
