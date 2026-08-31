/* Negative twin of rem_div_identity_equiv: the off-by-one identity is
 * falsifiable for every divisor, so the compositional remainder
 * lowering must still let the solver produce a counterexample. */
int nondet_int(void);

int main(void)
{
  int a = nondet_int();
  int b = nondet_int();
  if (b == 0)
    return 0;
  if (a == -2147483647 - 1 && b == -1)
    return 0;
  __ESBMC_assert(a % b == a - (a / b) * b + 1, "off-by-one identity");
  return 0;
}
