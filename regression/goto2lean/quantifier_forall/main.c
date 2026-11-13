unsigned nondet_uint(void);
int main(void)
{
  unsigned x = nondet_uint();
  unsigned i = 0, j = 0;
  __ESBMC_assume(x < 10);
  __ESBMC_assert(__ESBMC_forall(&i, i + x >= i || i > 4294967285u), "no wrap");
  __ESBMC_assert(
    __ESBMC_forall(&i, i <= 4294967295u) && __ESBMC_forall(&j, j >= x || j < 10),
    "conjunction");
  return 0;
}
