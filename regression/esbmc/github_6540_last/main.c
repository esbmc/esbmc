int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assert(x >= 0 || x < 0, "holds");
  __ESBMC_assert(x != 42, "may fail");
  return 0;
}
