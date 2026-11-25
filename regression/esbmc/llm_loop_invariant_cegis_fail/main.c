/* As llm_loop_invariant_cegis, with a genuinely false assertion: the model's
 * invariants are inductive and refute it, so it is reported FAILED rather than
 * masked or left unknown. */
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
