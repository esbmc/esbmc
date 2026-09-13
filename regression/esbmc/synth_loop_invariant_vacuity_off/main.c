/* --synthesise-loop-invariants turns the vacuity probe on for the whole run,
 * inheriting the coupling from --loop-invariant-check. It applies to every
 * claim, not only the ones a loop invariant reaches, so a program with no loop
 * at all can move from SUCCESSFUL to UNKNOWN. Documented in the option help;
 * pinned here so the coupling stays deliberate. */
int main(void)
{
  unsigned int n;
  __ESBMC_assume(n <= 4);
  __ESBMC_assume(n > 10);
  __ESBMC_assert(n == 0, "vacuous");
  return 0;
}
