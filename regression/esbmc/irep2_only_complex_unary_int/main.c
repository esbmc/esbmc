#include <assert.h>

/* The integral element type reaches the same unary arm as
   irep2_only_complex_unary's double one. Split from it because both element
   types in one formula take z3 past the Windows job's limit. */

int nondet_int(void);

int main(void)
{
  int _Complex a;
  __real__ a = nondet_int();
  __imag__ a = nondet_int();
  int ar = __real__ a, ai = __imag__ a;
  __ESBMC_assume(ar >= 1 && ar <= 4 && ai >= 1 && ai <= 4);

  int _Complex m = -a;
  assert(__real__ m == -ar && __imag__ m == -ai);
  int _Complex k = ~a;
  assert(__real__ k == ar && __imag__ k == -ai);

  return 0;
}
