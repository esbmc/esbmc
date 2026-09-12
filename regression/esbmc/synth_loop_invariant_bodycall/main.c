/* A user invariant reaches a callee through the loop *body*, not through the
 * marker's own expression. collect_marker_dependencies does not see that call,
 * so collect_annotated_loop_callees adds it: cutting f's loop would leave the
 * marker's preservation obligation to be discharged across a havoc-abstracted
 * f, the enclosed_by_user_invariant case reached through a call.
 *
 * synth_loop_invariant_bodycall_off is the same program without the marker. */
#include <assert.h>

static unsigned int f(unsigned int n)
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
  unsigned int o = 0;
  unsigned int total = 0;

  __ESBMC_loop_invariant(o <= 2 && total == 4 * o);
  while (o < 2)
  {
    total = total + f(4);
    o = o + 1;
  }

  assert(total == 8);
  return 0;
}
