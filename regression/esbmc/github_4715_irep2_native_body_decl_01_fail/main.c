// _fail sibling of github_4715_irep2_native_body_decl_01 (W1-loc spike Phase C,
// esbmc/esbmc#4715). Pins that consuming the trivial-type declarations natively
// (DECL + ASSIGN + scope-exit DEAD) neither corrupts the declared values nor
// suppresses bug detection: compute() still leaves g == 12, so the wrong-value
// assertion is a reachable violation reported as VERIFICATION FAILED under
// --irep2-native-body.
#include <assert.h>

int g;

void compute(int p)
{
  int y = p + 1;
  int z = y * 2;
  g = z;
}

int main(void)
{
  compute(5);
  assert(g == 13);
  return 0;
}
