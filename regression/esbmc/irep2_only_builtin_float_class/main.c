double nondet_double(void);

/* Each value is chosen so that the four predicates disagree on it: an infinity
   separates isinf from isnan and isfinite, and a zero separates isnormal from
   isfinite. A test over 1.0 alone cannot tell any of them apart. */
int main(void)
{
  double d = nondet_double(), z = nondet_double();
  __ESBMC_assume(d == 1.0 && z == 0.0);
  const double i = __builtin_inf();

  __ESBMC_assert(!__builtin_isnan(d), "1.0 is not NaN");
  __ESBMC_assert(!__builtin_isnan(i), "an infinity is not NaN");
  __ESBMC_assert(__builtin_isinf(i), "an infinity is infinite");
  __ESBMC_assert(!__builtin_isinf(d), "1.0 is not infinite");
  __ESBMC_assert(__builtin_isfinite(d), "1.0 is finite");
  __ESBMC_assert(!__builtin_isfinite(i), "an infinity is not finite");
  __ESBMC_assert(__builtin_isnormal(d), "1.0 is normal");
  __ESBMC_assert(!__builtin_isnormal(z), "0.0 is finite but not normal");
  return 0;
}
