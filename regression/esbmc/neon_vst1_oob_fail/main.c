// vst1q_f32 writes four floats: from o + 1 the last lane is one past the
// array.
#include <arm_neon.h>

int main(void)
{
  float o[4];
  vst1q_f32(o + 1, vdupq_n_f32(1.0f));
  return 0;
}
