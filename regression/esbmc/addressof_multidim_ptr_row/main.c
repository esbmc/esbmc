// Indexing through a pointer-to-pointer lowers to a dereference rather than an
// index chain, so the walk finds no enclosing row to linearise and leaves the
// address alone (#6778).
//
// __ESBMC_assert, not assert: MSVC spells assert as `(!!(e)) || (_wassert(..), 0)`,
// so an expression this fold makes constant-true short-circuits away before
// ESBMC sees it and no claim is generated at all on Windows.

int main(void)
{
  int r0[3] = {0, 1, 2};
  int r1[3] = {3, 4, 5};
  int *rows[2] = {r0, r1};
  int **pp = rows;

  __ESBMC_assert(&pp[1][2] == &r1[2], "&pp[1][2] == &r1[2]");
  return 0;
}
