double nondet_double(void);

int main(void)
{
  double a = nondet_double(), b = nondet_double();
  __ESBMC_assume(a == 2.0 && b == 1.0);

  __ESBMC_assert(__builtin_isgreater(a, b), "2 > 1");
  __ESBMC_assert(__builtin_isgreaterequal(a, b), "2 >= 1");
  __ESBMC_assert(!__builtin_isless(a, b), "2 is not < 1");
  __ESBMC_assert(__builtin_islessequal(b, b), "1 <= 1");
  __ESBMC_assert(__builtin_islessgreater(a, b), "2 and 1 differ");
  __ESBMC_assert(!__builtin_isunordered(a, b), "neither is NaN");
  return 0;
}
