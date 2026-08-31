/* Equivalence of `%` against the C99 6.5.5 defining identity
 * a == (a/b)*b + a%b. With the remainder encoded as its own solver
 * primitive this asks the solver to prove the rem and div circuits
 * pointwise equal — infeasible to bit-blast at 32 bits. Lowering the
 * remainder compositionally as a - (a/b)*b shares the division term
 * and the equivalence collapses structurally. */
int nondet_int(void);

int main(void)
{
  int a = nondet_int();
  int b = nondet_int();
  if (b == 0)
    return 0;
  if (a == -2147483647 - 1 && b == -1)
    return 0;
  __ESBMC_assert(a % b == a - (a / b) * b, "C99 6.5.5 identity");
  return 0;
}
