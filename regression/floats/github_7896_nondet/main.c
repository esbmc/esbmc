/* github #7896: a _Float16 that reaches the solver. The 4-bit exponent made
 * the sort (4, 12), which Bitwuzla rejects as an experimental format. */
_Float16 nondet_h(void);

int main(void)
{
  _Float16 x = nondet_h();
  __ESBMC_assume(x == (_Float16)1000.0f);
  __ESBMC_assert(x + x == (_Float16)2000.0f, "2000 is exact in binary16");
  return 0;
}
