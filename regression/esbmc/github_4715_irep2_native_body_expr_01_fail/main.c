// _fail sibling of github_4715_irep2_native_body_expr_01 (W1-loc spike Phase C,
// esbmc/esbmc#4715). Pins that consuming the side-effect-free expression
// statements natively as OTHER instructions neither corrupts the following
// assignment nor suppresses bug detection: g is still x+1 after touch(), so the
// wrong-value assertion is a reachable violation reported as VERIFICATION FAILED
// under --irep2-native-body.
#include <assert.h>

int g;

void touch(int x)
{
  x;
  g = x + 1;
}

void discard(int a, int b)
{
  (void)a;
  b;
}

int main(void)
{
  touch(5);
  assert(g == 7);

  discard(1, 2);
  return 0;
}
