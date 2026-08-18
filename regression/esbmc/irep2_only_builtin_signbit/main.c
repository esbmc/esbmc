double nondet_double(void);

int main(void)
{
  double p = nondet_double(), n = nondet_double();
  __ESBMC_assume(p == 1.0 && n == -1.0);

  __ESBMC_assert(!__builtin_signbit(p), "1.0 is positive");
  __ESBMC_assert(__builtin_signbit(n), "-1.0 is negative");
  return 0;
}
