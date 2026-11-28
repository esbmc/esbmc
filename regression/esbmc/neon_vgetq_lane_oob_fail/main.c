// An index read with vgetq_lane_u8 is 16, one past the end of the table.
#include <arm_neon.h>

int main(void)
{
  unsigned char table[16] = {0};
  uint8x16_t idx = vdupq_n_u8(15);
  idx = vsetq_lane_u8(16, idx, 3);
  return table[vgetq_lane_u8(idx, 3)];
}
