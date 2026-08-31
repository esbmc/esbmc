// A zero innermost index with a non-zero enclosing row: the offset lives
// entirely in the linearised rows, so the fold has to emit it even though
// &a[..][0] alone would not be worth rewriting (#6778).
//
// __ESBMC_assert, not assert: MSVC spells assert as `(!!(e)) || (_wassert(..), 0)`,
// so an expression this fold makes constant-true short-circuits away before
// ESBMC sees it and no claim is generated at all on Windows.

int main(void)
{
  int a[2][3];
  int *p = &a[1][0];
  int *q = &a[0][0] + 3;

  __ESBMC_assert(p == q, "p == q");
  return 0;
}
