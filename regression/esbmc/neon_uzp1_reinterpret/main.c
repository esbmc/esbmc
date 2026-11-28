// vuzp1_u8 is a __builtin_shufflevector, here of vreinterpret bit-casts,
// which symex took for constant vectors. The expected values are what the
// same code gives natively on AArch64.
#include <arm_neon.h>
#include <assert.h>

int main(void)
{
  uint16x4_t lo = {0x0102, 0x0304, 0x0506, 0x0708};
  uint16x4_t hi = vdup_n_u16(0x0a0b);
  uint8x8_t r = vuzp1_u8(vreinterpret_u8_u16(lo), vreinterpret_u8_u16(hi));
  assert(vget_lane_u8(r, 0) == 0x02 && r[1] == 0x04 && r[3] == 0x08);
  assert(r[4] == 0x0b && r[7] == 0x0b);
  return 0;
}
