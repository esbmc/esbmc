/* As llm_loop_invariant_cegis_fail, but the model answers with an invariant
 * that would mask the false assertion and does not hold on entry. The check
 * drops it, so the bug is never reported as SUCCESSFUL. */
#include <assert.h>

unsigned nondet_uint(void);

int main(void)
{
  unsigned n = nondet_uint();
  unsigned i = 0, x = 0, y = 0;

  while (i < n)
  {
    if (i % 2 == 0)
      x++;
    else
      y++;
    i++;
  }

  assert(x + y == n + 1);
  return 0;
}
