/* github #7929: the counterpart of the signed reads. An unsigned value with
 * the top bit set must keep its magnitude; reading it as two's complement
 * only shows up once the value reaches the simplifier, since an out-of-range
 * constant still prints correctly. */
unsigned int nondet_uint(void);

int main(void)
{
  unsigned int u = nondet_uint();
  __ESBMC_assume(u == (unsigned int)-4);
  unsigned int q = u / 3u;
  __ESBMC_assert(q == 9, "cex");
  return 0;
}
