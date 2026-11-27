// vld1q_u32 on a byte array inside a packed struct, at offset 0 and 1, is
// valid on AArch64, but ESBMC still reports "Incorrect alignment" because a
// packed object's base address is not assumed aligned.
#include <arm_neon.h>
#include <stdint.h>

struct __attribute__((packed)) S0
{
  uint8_t data[16];
  uint8_t tag;
};
struct __attribute__((packed)) S1
{
  uint8_t tag;
  uint8_t data[16];
};

int main(void)
{
  struct S0 s0 = {0};
  struct S1 s1 = {0};
  uint32x4_t a = vld1q_u32((const uint32_t *)s0.data);
  uint32x4_t b = vld1q_u32((const uint32_t *)s1.data);
  return a[0] + b[0];
}
