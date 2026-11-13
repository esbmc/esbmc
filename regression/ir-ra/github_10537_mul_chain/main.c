int nondet_int(void);
int main()
{
  int x = nondet_int();
  __ESBMC_assume(x >= 1 && x <= 3);
  int y = x * 65536 * 65536;
  __ESBMC_assert(y != 0, "2^16 * 2^16 is folded only if it fits");
  return 0;
}
