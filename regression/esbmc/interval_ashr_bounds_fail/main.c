int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x >= 16 && x <= 32);
  int y = x >> 2;
  __ESBMC_assert(y != 4, "x in [16,19] gives y == 4");
  return 0;
}
