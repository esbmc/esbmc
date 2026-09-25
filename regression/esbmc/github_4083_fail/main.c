// Negative counterpart of github_4083: the target is labels[1] (L2), so a
// claim that execution reaches L1 must be refuted rather than vacuously held
// -- this is what fails if the dispatch chain picks the wrong label or falls
// through (issue #4083).
int main()
{
  void *labels[] = {&&L1, &&L2};
  int x = 0;
  goto *labels[1];
L1:
  x = 1;
  goto END;
L2:
  x = 2;
  goto END;
END:
  __ESBMC_assert(x == 1, "must not reach L1");
  return 0;
}
