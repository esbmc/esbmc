unsigned nondet_uint(void);
int main(void)
{
  unsigned x = nondet_uint();
  unsigned long s = sizeof((x < 10) ? 1 : 0);
  unsigned long b = sizeof(_BitInt(17));
  __ESBMC_assert(s == 4 && b == 4, "clang sizes");
  return 0;
}
