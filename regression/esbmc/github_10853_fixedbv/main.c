extern float __VERIFIER_nondet_float(void);
int main(void)
{
  float a = __VERIFIER_nondet_float(), b = __VERIFIER_nondet_float();
  unsigned i = 0;
  __ESBMC_loop_invariant(i <= 3 && a + b - b == a);
  while (i < 3)
    i++;
  return 0;
}
