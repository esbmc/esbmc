unsigned nondet_uint(void);
int main(void)
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n > 0 && n < 4);
  return (int)sizeof(int[n]);
}
