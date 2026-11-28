// An index read with vgetq_lane_u8 is 15, the last element of the table. It
// used to be nondet, so the access was reported out of bounds.
#include <arm_neon.h>

int main(void)
{
  unsigned char table[16] = {0};
  uint8x16_t idx = vdupq_n_u8(15);
  idx = vsetq_lane_u8(4, idx, 3);
  return table[vgetq_lane_u8(idx, 3)] + table[vgetq_lane_u8(idx, 0)];
}
