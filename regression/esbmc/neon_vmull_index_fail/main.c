// The index vmull_u8 computes is 16 * 16 = 256, one past the end of table.
#include <arm_neon.h>

int main(void)
{
  unsigned char table[256] = {0};
  uint16x8_t i = vmull_u8(vdup_n_u8(16), vdup_n_u8(16));
  return table[vgetq_lane_u16(i, 0)];
}
