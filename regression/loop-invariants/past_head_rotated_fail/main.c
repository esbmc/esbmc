/* A rotated loop is entered at its test, past its head. Unwound, the false
 * invariant x == 0 cannot hide the assignment at i == 5, so the assertion
 * fails. */
#include <assert.h>
int main()
{
  int i = 0, x = 0;
  __ESBMC_loop_invariant(i >= 0 && i <= 10 && x == 0);
  goto test;
body:
  i = i + 1;
  if (i == 5)
    x = 1;
test:
  if (i < 10)
    goto body;
  assert(x == 0);
  return 0;
}
