/* Equivalence of `%` against the C99 6.5.5 defining identity
 * a == (a/b)*b + a%b, at 64 bits. With the remainder encoded as its
 * own solver primitive this asks the solver to prove the rem and div
 * circuits pointwise equal — infeasible to bit-blast already at 32
 * bits and hopeless at 64, so an unfixed engine exceeds any CI budget
 * by hours, not seconds. Lowering the remainder compositionally as
 * a - (a/b)*b shares the division term and the equivalence collapses
 * structurally, solving in milliseconds. */
long long nondet_i8(void);

int main(void)
{
  long long a = nondet_i8();
  long long b = nondet_i8();
  if (b == 0)
    return 0;
  if (a == -9223372036854775807LL - 1 && b == -1)
    return 0;
  __ESBMC_assert(a % b == a - (a / b) * b, "C99 6.5.5 identity");
  return 0;
}
