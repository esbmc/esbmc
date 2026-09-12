/* The SV-COMP sum01 shape: signed counter, signed accumulator, literal addend.
 * Unreachable before signed support -- the two-disjunct bound is false at entry
 * when the bound may be negative. With a literal addend there is no symbolic
 * multiplier, so the third disjunct i == i0 is affordable and establishment
 * becomes unconditional. */
#define a (2)
int nondet_int();
int main()
{
  int i, n = nondet_int(), sn = 0;
  for (i = 1; i <= n; i++)
    sn = sn + a;
  __ESBMC_assert(sn == n * a || sn == 0, "sum01");
  return 0;
}
