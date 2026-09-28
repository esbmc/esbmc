/* A nondeterministic dividend keeps the division out of the simplifier, so
   the quotient's VCC actually reaches the solver rather than passing
   vacuously (#6713). */
float nondet_float(void);

int main(void)
{
  float x = nondet_float();
  __ESBMC_assume(x > 2.0f && x < 100.0f);

  _Complex float a = x, b = 2.0f;
  a /= b;

  __ESBMC_assert(__real__ a > 1.0f, "quotient stays above one");
  return 0;
}
