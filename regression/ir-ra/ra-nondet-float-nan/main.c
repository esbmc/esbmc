extern float __VERIFIER_nondet_float(void);

int main(void)
{
  /* Single-precision nondet float also gets a NaN predicate. */
  float x = __VERIFIER_nondet_float();

  __ESBMC_assert(x == x, "unconstrained nondet float may be NaN");
  return 0;
}
