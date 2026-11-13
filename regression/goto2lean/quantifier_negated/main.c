unsigned nondet_uint(void);
int main(void)
{
  unsigned x = nondet_uint();
  unsigned i = 0;
  __ESBMC_assert(!__ESBMC_forall(&i, i < x), "some i is not below x");
  return 0;
}
