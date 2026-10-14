/* Negative pair of synth_loop_invariant_calleeinv: same shape, but the user's
 * invariant asserts the wrong value for the call. Synthesis must still stand
 * down inside count_to, and the wrong invariant must still be reported FAILED
 * rather than downgraded by a havoc this pass had no business introducing. */
#include <assert.h>

static unsigned int count_to(unsigned int n)
{
  unsigned int i = 0;
  unsigned int s = 0;
  while (i < n)
  {
    s = s + 1;
    i = i + 1;
  }
  return s;
}

int main(void)
{
  unsigned int i = 0;

  __ESBMC_loop_invariant(i <= 4);
  __ESBMC_loop_invariant(count_to(4) == 5);
  while (i < 4)
    i++;

  assert(i == 4);
  return 0;
}
