/* Regression: GitHub #7480 -- an inner loop's base case is emitted inside the
 * outer body, downstream of the outer havoc, so the state it is checked from
 * is arbitrary rather than concrete.  `s == i` holds every time the real
 * program enters the inner loop and is still refuted there, so the base case
 * cannot be exempted from the downgrade by name; only the outermost one, which
 * no havoc precedes, is decided against the program. */
#include <assert.h>

int main()
{
  int i = 0, s = 0;
  __ESBMC_loop_invariant(i >= 0);
  while (i < 3)
  {
    int j = 0;
    __ESBMC_loop_invariant(j >= 0 && s == i);
    while (j < 2)
    {
      j++;
    }
    s++;
    i++;
  }
  return 0;
}
