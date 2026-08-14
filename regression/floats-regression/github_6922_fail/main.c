#include <assert.h>

/* Negative twin: the store selects 7.5f, so asserting 2.5f must be reported.
 * A store that yields an unconstrained value would let this through. */
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
  assert(y[0] == 2.5f);
  return 0;
}
