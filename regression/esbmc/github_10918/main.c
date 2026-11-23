/* #10918: no object crosses 2^63, so the addresses inside one keep their
 * order and difference as intptr_t. */
#include <assert.h>
#include <stdint.h>
int nondet_int(void);
int main(void)
{
  uint64_t g[4];
  int i = nondet_int(), j = nondet_int();
  __ESBMC_assume(0 <= i && i < 4 && 0 <= j && j < 4);
  intptr_t a = (intptr_t)&g[i], b = (intptr_t)&g[j];
  intptr_t r;
  assert((intptr_t)&g[0] <= (intptr_t)&g[3]);
  assert(!__builtin_sub_overflow(a, b, &r) && r == (i - j) * 8);
  return 0;
}
