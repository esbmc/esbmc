extern float __VERIFIER_nondet_float(void);
int main()
{
  float x = __VERIFIER_nondet_float();
  __ESBMC_assume(x >= 0.0f && x <= 1.0f);
  __ESBMC_assert(x <= 2.0f, "x stays in range");
  return 0;
}
