/* #9785: a pointer made from an integer that may point into either array
 * takes its offset from the integer within each; checked with pointer checks
 * on. */
#include <stdint.h>
#include <assert.h>
_Bool nondet_bool(void); unsigned nondet_uint(void);
int a[4], b[4];
int main(void) {
  _Bool c = nondet_bool(); unsigned k = nondet_uint();
  __ESBMC_assume(k < 4);
  uintptr_t base = c ? (uintptr_t)a : (uintptr_t)b;
  int *p = (int *)(base + 4 * k);
  *p = 7;
  assert(c ? a[k] == 7 : b[k] == 7);
  assert(a[0] != 7);
  return 0;
}
