// _fail sibling of github_4715_irep2_native_body_call_01 (W1-loc spike
// Phase C, esbmc/esbmc#4715). Pins that a genuine violation through native
// call statements is still reported as VERIFICATION FAILED, not silently
// dropped, under --irep2-native-body: run() returns 5, so the wrong-value
// assertion is a reachable violation.
#include <assert.h>

int g;

void bump(int x)
{
  g = g + x;
}

void reset(void)
{
  g = 0;
}

void bump_twice(int x, int y)
{
  bump(x);
  bump(y);
}

int run(void)
{
  reset();
  bump_twice(2, 3);
  return g;
}

int main(void)
{
  assert(run() == 6);
  return 0;
}
