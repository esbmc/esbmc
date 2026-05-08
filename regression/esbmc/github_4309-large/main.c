#include <assert.h>

// Stress test: 2 witnesses (x = -1, x = 1) with a large goto trace.
// The unrolled loop inflates the per-witness state count without
// increasing the witness count. Used to characterise the scaling
// behaviour of the multi-witness pretty-printer.

int main(void)
{
  int x;
  if (x > 0)
    x--;
  else
    x++;

  // Padding: forces the trace to contain many derived states.
  // With --unwind 51 the loop fully unrolls (~50 iterations × a few
  // states each). The padding work is independent of x, so the diff
  // between the two witnesses is concentrated in the if-branch.
  int s = 0;
  for (int i = 0; i < 50; i++)
    s += i;

  assert(x != 0 || s != 1225);
  return 0;
}
