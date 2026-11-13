unsigned nondet_uint(void);

int main(void)
{
  unsigned n = nondet_uint();
  __ESBMC_assert(n >= 0, "an unsigned is never negative");
  return 0;
}
