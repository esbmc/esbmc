int nondet_int(void);

/* A 12-bit x % m spelled as x - (x / m) * m. Bitwuzla cannot settle it by
 * propagation, so CaDiCaL runs a local-search round, and that round sizes its
 * score table with a loop that never ends if the FPU rounds upward. */
int main(void)
{
  _BitInt(12) x = (_BitInt(12))nondet_int();
  _BitInt(12) m = (_BitInt(12))nondet_int();
  __ESBMC_assume(x >= 0 && m > 0);
  _BitInt(12) r = x - (x / m) * m;
  __ESBMC_assert(r > 0, "remainder is positive");
  return 0;
}
