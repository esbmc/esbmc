#include <assert.h>

/* The loop bound arrives as a parameter, so the index is symbolic and the
 * store is not folded to a constant offset. A whole float element written
 * through a symbolic index used to be decomposed into four byte updates;
 * each byte round-tripped the element through fp.to_ieee_bv / to_fp, and
 * SMT-LIB leaves the NaN pattern fp.to_ieee_bv returns unconstrained, so the
 * value read back need not be the one stored. */
static void st(const float *x, float *y, int n)
{
  int k;
  for (k = 0; k < n - 1; k++)
    y[k] = (x[k] < x[k + 1]) ? 7.5f : 2.5f;
}

int main(void)
{
  float x[2], y[1];
  x[0] = 0.25f;
  x[1] = 0.75f;
  st(x, y, 2);
  assert(y[0] == 7.5f); /* 0.25f < 0.75f */
  return 0;
}
