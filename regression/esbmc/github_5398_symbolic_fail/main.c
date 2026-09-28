// With a symbolic size the guard stays a genuine disjunction rather than
// folding away, so the zero-sized outcome must still be explored: writing
// through the success arm is out of bounds when the request was for 0 bytes.
#include <stdlib.h>

extern unsigned long __VERIFIER_nondet_ulong(void);
extern void __VERIFIER_assume(int);

int main(void)
{
  unsigned long n = __VERIFIER_nondet_ulong();
  __VERIFIER_assume(n <= 4);

  char *p = malloc(n);
  __VERIFIER_assume((unsigned long)p != (unsigned long)0);
  p[0] = 1;
  return 0;
}
