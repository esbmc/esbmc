/* Pins the cap on the symbolic-size arm, which the constant-size tests cannot
   reach. The window matters: every other symbolic-size test sits at or above
   0xFFFFFFFFFFFFFF00, which exceeds the pre-cap bound too, so none of them
   discriminates PTRDIFF_MAX from it. Deliberately NOT --force-malloc-success:
   under that flag a symbolic size is bounded only for layability (R38). */
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

size_t nondet_size(void);

int main(void)
{
  size_t n = nondet_size();
  __ESBMC_assume(n > (size_t)PTRDIFF_MAX && n <= 0xFFFFFFFFFFFFFF00UL);

  char *b = malloc(n);
  assert(b == NULL);
  return 0;
}
