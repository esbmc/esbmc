// _fail sibling of github_4715_irep2_native_body_ifthenelse_01 (W1-loc spike
// Phase C, esbmc/esbmc#4715). Pins that consuming if/else natively neither
// corrupts branch results nor suppresses bug detection: classify(-5) is -1, so
// the wrong-value assertion is a reachable violation reported as VERIFICATION
// FAILED under --irep2-native-body.
#include <assert.h>

int classify(int x)
{
  if (x < 0)
    return -1;
  else
    return 1;
}

int main(void)
{
  assert(classify(-5) == 1);
  return 0;
}
