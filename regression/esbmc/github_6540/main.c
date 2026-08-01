int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assert(x != 42, "may fail");
  __ESBMC_assert(x >= 0 || x < 0, "holds");
  return 0;
}
