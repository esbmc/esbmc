// LD1 and ST1 have no alignment requirement: vld1q_u32 and vst1q_u32 on
// buf + 1 are valid and move the bytes there in little-endian lanes.
#include <arm_neon.h>
#include <assert.h>
#include <stdint.h>

int main(void)
{
  uint8_t buf[17];
  for (int i = 0; i < 17; i++)
    buf[i] = i;
  uint32x4_t v = vld1q_u32((const uint32_t *)(buf + 1));
  assert(v[0] == 0x04030201);
  assert(v[3] == 0x100f0e0d);
  vst1q_u32((uint32_t *)(buf + 1), vaddq_u32(v, vdupq_n_u32(0x10101010)));
  assert(buf[0] == 0);
  assert(buf[1] == 0x11);
  assert(buf[16] == 0x20);
  return 0;
}
