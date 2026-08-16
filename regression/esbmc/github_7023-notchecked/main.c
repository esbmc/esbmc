int main()
{
  int a, b;

  __ESBMC_assert(a + b == b + a, "addition commutes");
  /* At most one of these can be violated by any single model, so exactly one
     stays undecided whichever one the solver picks. */
  __ESBMC_assert(a != 1, "a is not one");
  __ESBMC_assert(a != 2, "a is not two");
  return 0;
}
