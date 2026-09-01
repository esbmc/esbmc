/* The loop-invariant schema cuts the loop, so the code after it is reached
 * only through the havoc-and-assume path. If that path is over-constrained --
 * or if a claim is wrongly carried over from an earlier Houdini round -- every
 * post-loop claim discharges without ever being solved. assert(0) is reachable
 * here, so anything but a violation means the pipeline stopped checking. */
#include <assert.h>

int main()
{
  float x = 2;

  while (nondet_bool())
    x = 2 * x - 1;

  assert(0);
  return 0;
}
