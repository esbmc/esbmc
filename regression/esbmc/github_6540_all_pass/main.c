int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assert(x >= 0 || x < 0, "holds one");
  __ESBMC_assert(x == x, "holds two");
  return 0;
}
