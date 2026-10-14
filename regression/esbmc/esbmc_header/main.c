// include/esbmc.h is the supported spelling of the verification intrinsics.
// Each macro must reach the __ESBMC_* it stands for, so the assumption below
// has to constrain the assertion and the allocation has to be a real one. #4610
#include <esbmc.h>

int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  ESBMC_assume(x > 10 && x < 1000);
  ESBMC_assert(x > 5, "the assumption carries into the assertion");

  int *p = (int *)ESBMC_alloca(sizeof(int));
  *p = x;
  ESBMC_assert(*p > 10, "alloca returns writable storage");
  ESBMC_assert(ESBMC_same_object(p, p), "a pointer shares its own object");

  ESBMC_atomic_begin();
  x++;
  ESBMC_atomic_end();
  ESBMC_yield();
  ESBMC_assert(x > 11, "the atomic region still updates x");

  ESBMC_unroll(2);
  return 0;
}
