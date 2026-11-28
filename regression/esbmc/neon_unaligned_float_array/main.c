// A float lane at a misaligned offset inside a float array is read through
// the packed one-member struct, which ESBMC rejects with "Object accessed
// with incompatible base type" (see #11135).
#include <arm_neon.h>
#include <stdint.h>

int main(void)
{
  float g[5] = {0};
  float32x4_t v = vld1q_f32((const float *)((uint8_t *)g + 2));
  return 0;
}
