/* Equivalence of `%` against the C99 6.5.5 defining identity
 * a == (a/b)*b + a%b, at 64 bits. With the remainder encoded as its
 * own solver primitive this asks the solver to prove the rem and div
 * circuits pointwise equal — infeasible to bit-blast already at 32
 * bits and hopeless at 64, so an unfixed engine exceeds any CI budget
 * by hours, not seconds. Lowering the remainder compositionally as
 * a - (a/b)*b shares the division term and the equivalence collapses
 * structurally, solving in milliseconds. */
long long nondet_ll(void);

int main(void)
{
  long long a = nondet_ll();
  long long b = nondet_ll();
  if (b == 0)
    return 0;
  if (a == -9223372036854775807LL - 1 && b == -1)
    return 0;
  __ESBMC_assert(a % b == a - (a / b) * b, "C99 6.5.5 identity");
  /* Concrete pins for C's TRUNCATING division: the identity above
   * holds for any consistent div/rem pair, flooring included, so these
   * anchor the sign convention itself. */
  __ESBMC_assert(-7 % 3 == -1, "truncated remainder, negative dividend");
  __ESBMC_assert(7 % -3 == 1, "truncated remainder, negative divisor");
  __ESBMC_assert(-7 / 3 == -2, "truncated quotient");
  return 0;
}
