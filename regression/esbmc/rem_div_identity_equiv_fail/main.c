/* Negative twin of rem_div_identity_equiv: the off-by-one identity is
 * falsifiable for every divisor, so the compositional remainder
 * lowering must still let the solver produce a counterexample. */
long long nondet_ll(void);

int main(void)
{
  long long a = nondet_ll();
  long long b = nondet_ll();
  if (b == 0)
    return 0;
  if (a == -9223372036854775807LL - 1 && b == -1)
    return 0;
  __ESBMC_assert(a % b == a - (a / b) * b + 1, "off-by-one identity");
  return 0;
}
