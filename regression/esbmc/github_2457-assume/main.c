#include <assert.h>
#include <stdlib.h>

/* Each assumption below is false under the real memory model, so main is
   unreachable. Were the __CPROVER_* primitives havoc'd, every assumption would
   be satisfiable and the assert(0) reachable. */
int main()
{
  char a[4];
  char *d = malloc(8);
  __ESBMC_assume(__CPROVER_r_ok(a, 8));
  __ESBMC_assume(__CPROVER_w_ok(a + 3, 2));
  __ESBMC_assume(__CPROVER_OBJECT_SIZE(a) != 4);
  __ESBMC_assume(__CPROVER_DYNAMIC_OBJECT(a));
  __ESBMC_assume(!__CPROVER_same_object(a, a + 1));
  __ESBMC_assume(!__CPROVER_DYNAMIC_OBJECT(d));
  assert(0);
  return 0;
}
