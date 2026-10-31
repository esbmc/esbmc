extern unsigned long nondet_ulong(void);

int main(void)
{
  unsigned long u = nondet_ulong();
  __ESBMC_assume(u >= 9223372036854775808UL && u <= 18446744073709551615UL);
  long s = (long)u;
  __ESBMC_assert(s < 0, "an unsigned value above the signed maximum reads back negative");
  return 0;
}
