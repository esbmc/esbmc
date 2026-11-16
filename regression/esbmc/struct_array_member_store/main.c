#include <stdint.h>
unsigned int __VERIFIER_nondet_uint(void);
typedef struct { uint32_t f0, f1, f2, f3, f4, f5, f6, f7; uint32_t shadow[16]; } state_t;
static state_t s;
int main(void)
{
  s.f3 = 7u;
  uint32_t k = __VERIFIER_nondet_uint();
  __ESBMC_assume(k < 16u);
  uint32_t *p = &s.shadow[k];
  *p = 5u;
  __ESBMC_assert(s.shadow[k] == 5u, "store lands at shadow[k]");
  __ESBMC_assert(s.f3 == 7u, "f3 untouched");
  return 0;
}
