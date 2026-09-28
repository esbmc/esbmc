int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x >= 16 && x <= 32);
  int y = x >> 2;
  __ESBMC_assert(y >= 4 && y <= 8, "x >> 2 is in [4,8] for x in [16,32]");
  return 0;
}
