int nondet_int(void);

int main(void)
{
  int i = nondet_int();
  int k = nondet_int();
  int m = nondet_int();
  __ESBMC_assume(i >= 200000);
  __ESBMC_assume(k >= 300000);
  __ESBMC_assume(m <= -400000);
  __ESBMC_assert(i >= 0 && k >= 0 && m <= 0, "each bound prints on its own line");
  return 0;
}
