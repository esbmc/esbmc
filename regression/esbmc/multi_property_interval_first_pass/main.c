/* esbmc/esbmc#7900: __ESBMC_assert(0, ...) reaches the first interval pass as
   a literal ASSERT 0, and that pass treated it as the end of the path: x = 1
   was never seen, and w was checked as though x were always 0. */
int main()
{
  int x = 0;
  if (nondet_int())
  {
    __ESBMC_assert(0, "v");
    x = 1;
    x = x + 1;
  }
  __ESBMC_assert(x == 0, "w");
  return 0;
}
