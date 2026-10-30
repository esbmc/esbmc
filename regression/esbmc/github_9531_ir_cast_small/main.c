extern unsigned long nondet_ulong(void);

int main(void)
{
  unsigned long u = nondet_ulong();
  __ESBMC_assume(u >= 5 && u <= 7);
  long s = (long)u;
  __ESBMC_assert(s >= 5 && s <= 7, "a value in range is kept");
  return 0;
}
