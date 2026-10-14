int nondet_int();

/* Branches hold two assertions each so goto_convert's guarded-assertion
   folding (which only fires on a single assert(false) branch body) does not
   apply: the asserts must stay behind the branch for the unreached-claims
   listing to observe their reachability. */
int main()
{
  if (3 > 0)
  {
    __ESBMC_assert(1, "reachable concrete then");
  }
  else
  {
    __ESBMC_assert(0, "dead concrete else A");
    __ESBMC_assert(1, "dead concrete else B");
  }

  int a = nondet_int();
  if (a % 2)
    ++a;
  __ESBMC_assert(a % 2 == 0, "a is even");

  if (a % 2)
  {
    __ESBMC_assert(0, "dead symbolic then A");
    __ESBMC_assert(1, "dead symbolic then B");
  }
  else
  {
    __ESBMC_assert(1, "reachable symbolic else");
  }
}
