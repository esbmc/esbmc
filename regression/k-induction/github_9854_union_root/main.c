#include <stdint.h>
unsigned int __VERIFIER_nondet_uint(void);
typedef union { uint32_t word; uint8_t byte; } reg_t;
static reg_t g_reg;
int main(void)
{
  uint32_t n = __VERIFIER_nondet_uint();
  __ESBMC_assume(n <= 1000000u);
  g_reg.word = 0x100u;
  for (uint32_t i = 0; i < n; i++)
  {
    g_reg.byte = (uint8_t)__VERIFIER_nondet_uint();
    __ESBMC_assert(g_reg.word >= 0x100u, "upper bytes kept");
  }
  return 0;
}
