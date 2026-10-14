extern float __VERIFIER_nondet_float(void);

int main(void)
{
  float x = __VERIFIER_nondet_float();
  /* Any nonzero x with |x| < 2^-74.5 has x*x < 2^-149 (float min subnormal),
   * so IEEE 754 rounds x*x to zero. */
  __ESBMC_assume(x > 0.0f && x < 1e-22f);
  float y = x * x;
  __ESBMC_assert(y != 0.0f, "float mul of tiny x underflows to zero");
  return 0;
}
