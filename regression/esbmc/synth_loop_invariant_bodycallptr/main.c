/* synth_loop_invariant_bodycall through a function pointer: the annotated
 * loop's body reaches its callee indirectly, so collect_annotated_loop_callees
 * cannot name the function whose loops must be left alone. Protecting one
 * callee is no longer an option, so the pass stands down for the whole program
 * rather than cut a loop the marker's preservation obligation reads through. */
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

static unsigned int (*fp)(unsigned int) = f;

int main(void)
{
  unsigned int o = 0;
  unsigned int total = 0;

  __ESBMC_loop_invariant(o <= 2 && total == 4 * o);
  while (o < 2)
  {
    total = total + fp(4);
    o = o + 1;
  }

  assert(total == 8);
  return 0;
}
