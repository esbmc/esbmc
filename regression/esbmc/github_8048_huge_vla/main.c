/* #8048: under --no-vla-size-check symex admits an object larger than
 * PTRDIFF_MAX, so it keeps an exact layout and an address round-trips
 * through uintptr_t (C11 7.20.1.4p1). */
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
  return 0;
}
