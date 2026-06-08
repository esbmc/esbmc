// `for`-init declaration shadows an outer capture `i`.  The outer `i`
// is used at depth 1 before the `for` (so it must be captured), but the
// loop's own `i` must not be rewritten.
int main()
{
  int i = 5;
  int inner(void)
  {
    int r = i;
    for (int i = 0; i < 3; ++i)
      r += i;
    return r;
  }
  __ESBMC_assert(inner() == 8, "for-init shadow preserved");
  return 0;
}
