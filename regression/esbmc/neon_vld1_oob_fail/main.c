// vld1q_u8 reads 16 bytes: from a + 1 the last lane is one past the array.
#include <arm_neon.h>
#include <stdint.h>

int main(void)
{
  uint8_t a[16] = {0};
  uint8x16_t v = vld1q_u8(a + 1);
  return v[0];
}
