// _fail sibling of github_4715_irep2_native_body_return_01 (W1-loc spike Phase C,
// esbmc/esbmc#4715). Pins that consuming the value-returns natively (RETURN + the
// end-of-function GOTO, with the trailing-goto block guard suppressing the local's
// DEAD) neither corrupts the returned values nor suppresses bug detection:
// inc(5)+doubled(6) is 18, so the wrong-value assertion is a reachable violation
// reported as VERIFICATION FAILED under --irep2-native-body.
#include <assert.h>

int inc(int p)
{
  return p + 1;
}

int doubled(int p)
{
  int t = p * 2;
  return t;
}

int main(void)
{
  int a = inc(5);
  int b = doubled(6);
  assert(a + b == 19);
  return 0;
}
