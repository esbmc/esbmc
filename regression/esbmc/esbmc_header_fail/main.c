// ESBMC_assert must be the real assertion, not a no-op that silently passes:
// x is unconstrained, so this has to be refutable. #4610
#include <esbmc.h>

int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  ESBMC_assert(x > 5, "unconstrained value is not bounded");
  return 0;
}
