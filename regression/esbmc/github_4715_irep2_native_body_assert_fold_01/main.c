// generate_ifthenelse folds a branch that reduces to a lone assert(false) into
// the guard rather than emitting a conditional GOTO. The native if-arm used to
// decline these shapes, taking the whole function to the round-trip; it now
// reproduces the fold (W1-loc, esbmc/esbmc#4715). One case per fold branch:
// then-only, then-only in a block, else-only, both, and then-only with a
// no-op else -- the last is why the fold tests the branch program, not the AST.
extern int nd(void);
int main(void)
{
  int c = nd();
  if (c == 1)
    __ESBMC_assert(0, "a");
  if (c == 2)
  {
    __ESBMC_assert(0, "b");
  }
  if (c == 3)
    ;
  else
    __ESBMC_assert(0, "c");
  if (c == 4)
    __ESBMC_assert(0, "d");
  else
    __ESBMC_assert(0, "e");
  if (c == 5)
    __ESBMC_assert(0, "f");
  else
    ;
  return 0;
}
