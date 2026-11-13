#include <limits.h>
extern int __VERIFIER_nondet_int(void);
int main(void)
{
  int x = __VERIFIER_nondet_int();
  __ESBMC_assume(x >= 0 && x < INT_MAX);
  x += 1;
  __ESBMC_assert(0, "ignored user assertion");
  return 0;
}
