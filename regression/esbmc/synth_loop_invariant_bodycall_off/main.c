/* The mutation partner of synth_loop_invariant_bodycall: without the marker
 * nothing protects f, and its loop is synthesised. */
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

  while (o < 2)
  {
    total = total + f(4);
    o = o + 1;
  }

  assert(total == 8);
  return 0;
}
