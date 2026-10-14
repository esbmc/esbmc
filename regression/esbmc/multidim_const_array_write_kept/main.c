// Two writes into one row of a multi-dimensional array, under a guard. With
// the array propagated, both land in a single `A WITH [0 := A[0] WITH [0:=1]
// WITH [1:=2]]`, and the SMT flattening in decompose_store_chain() follows
// only the update-value spine: it drops the first write and this verifies as
// a violated assertion. Multi-dimensional writes therefore stay off the
// propagation path (R42).
#include <assert.h>

int main(void)
{
  int r[2][2];
  _Bool c = nondet_bool();

  __ESBMC_assume(c);
  if (c)
  {
    r[0][0] = 1;
    r[0][1] = 2;
  }

  assert(r[0][0] == 1);
  assert(r[0][1] == 2);
  return 0;
}
