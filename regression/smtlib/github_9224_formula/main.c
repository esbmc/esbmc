int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assert(x / 4 + x % 8 != 12345, "");
  return 0;
}
