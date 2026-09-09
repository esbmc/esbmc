/* A user invariant on the outer loop protects the inner one too.
 *
 * has_user_invariant only sees the ten instructions ahead of its own head, so
 * the outer marker is neither in range nor reachable from the inner loop's
 * head. Cutting the inner loop would leave the outer marker's preservation
 * obligation to be discharged across a body containing a havoc -- the same
 * mechanism collect_invariant_dependencies rules out for a callee.
 *
 * synth_loop_invariant_nestedinv_off is the same program without the outer
 * marker; there the inner loop is synthesised, so this test's silence is the
 * guard and not the recogniser declining for some other reason. */
#include <assert.h>

int main(void)
{
  unsigned int o = 0;
  unsigned int total = 0;

  __ESBMC_loop_invariant(o <= 2 && total == 4 * o);
  while (o < 2)
  {
    unsigned int i = 0;
    unsigned int s = 0;
    while (i < 4)
    {
      s = s + 1;
      i = i + 1;
    }
    total = total + s;
    o = o + 1;
  }

  assert(total == 8);
  return 0;
}
