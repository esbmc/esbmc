// #7891: __bf16 arithmetic reaches the solver in bfloat16's format.
#include <assert.h>

__bf16 nondet_bf16(void);

int main(void)
{
  __bf16 x = nondet_bf16();
  __ESBMC_assume((float)x == 128.0f);
  // 128.5 needs 9 significand bits: it ties to even in bfloat16's 8.
  __bf16 y = x + (__bf16)0.5f;
  assert((float)y == 128.5f);
  return 0;
}
