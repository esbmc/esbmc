// Under --ir integers are unbounded, so ~b must be -b - 1 for every b, not
// the 64-bit complement (#10596). --no-simplify keeps the simplifier from
// rewriting ~b + 1 as -b before the solver sees it.
unsigned nondet_uint(void);
long long nondet_longlong(void);

int main(void)
{
  unsigned b = nondet_uint();
  __ESBMC_assume(b <= 10);
  unsigned y = ~b + 1u;
  __ESBMC_assert(y + b == 0, "~b + 1 == -b");

  long long x = nondet_longlong();
  __ESBMC_assert(~x == -x - 1, "~x == -x - 1");
  return 0;
}
