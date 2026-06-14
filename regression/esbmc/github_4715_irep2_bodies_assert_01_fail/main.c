// esbmc/esbmc#4715 (V.4.4 parity): negative variant of assert_01. The libc
// `assert` ternary still lowers under --irep2-bodies after the migrate if-arm
// coerces its mismatched (void/int) branches, and -- crucially -- still catches
// a genuine violation. The assertion below is false, so ESBMC must report
// VERIFICATION FAILED rather than aborting or passing vacuously.
#include <assert.h>

int main()
{
  int x = 5;
  assert(x == 6);
  return 0;
}
