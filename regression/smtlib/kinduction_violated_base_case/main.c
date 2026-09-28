unsigned nondet_uint(void);

int main(void)
{
  unsigned x = nondet_uint();
  __ESBMC_assert(x != 4242u, "x is not 4242");
  return 0;
}
