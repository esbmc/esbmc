double nondet_double(void);
double sqrt(double);

int main(void)
{
  double d = nondet_double();
  __ESBMC_assume(d == -3.0);

  __ESBMC_assert(__builtin_fabs(d) == 3.0, "fabs negates a negative");
  __ESBMC_assert(sqrt(d * d) == 3.0, "sqrt of 9 is 3");
  __ESBMC_assert(__builtin_isinf(__builtin_inf()), "inf is infinite");
  __ESBMC_assert(__builtin_huge_val() > d, "huge_val exceeds any finite value");
  return 0;
}
