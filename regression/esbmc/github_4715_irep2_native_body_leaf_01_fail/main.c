// _fail sibling of github_4715_irep2_native_body_leaf_01 (W1-loc spike Phase C,
// esbmc/esbmc#4715). Pins that consuming the decl-free bodies natively does not
// suppress bug detection: the reachable assertion violation in main() must
// still be reported as VERIFICATION FAILED under --irep2-native-body.
#include <assert.h>

void nop(void)
{
  ;
}

void noop(void)
{
}

int main(void)
{
  nop();
  noop();
  int x = 2;
  assert(x == 3);
  return 0;
}
