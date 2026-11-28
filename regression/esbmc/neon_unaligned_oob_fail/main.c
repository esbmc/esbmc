// vld1q_u32 reads 16 bytes: from buf + 2 the last lane ends one byte past
// the array, so the check must still fire on an unaligned address.
#include <arm_neon.h>
#include <stdint.h>

int main(void)
{
  uint8_t buf[17] = {0};
  uint32x4_t v = vld1q_u32((const uint32_t *)(buf + 2));
  return v[0];
}
