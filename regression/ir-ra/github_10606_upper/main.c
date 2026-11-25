unsigned nondet_uint(void);

int main(void)
{
  unsigned n = nondet_uint();
  __ESBMC_assert(n <= 4294967295u, "an unsigned fits in 32 bits");
  return 0;
}
