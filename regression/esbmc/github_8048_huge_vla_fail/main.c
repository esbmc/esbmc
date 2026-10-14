/* #8048: an object larger than PTRDIFF_MAX that symex admits under
 * --no-vla-size-check can still be laid out. */
#include <assert.h>
#include <stdint.h>
unsigned __VERIFIER_nondet_uint(void);
int main()
{
  unsigned n = __VERIFIER_nondet_uint();
  __ESBMC_assume(n > 2147483648u && n < 4000000000u);
  char a[n];
  char *p = a + 3;
  assert((char *)(uintptr_t)p == p);
  assert(n < 3000000000u);
  return 0;
}
