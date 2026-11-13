unsigned nondet_uint(void);
int main(void) {
  unsigned x = nondet_uint();
  unsigned i = 0;
  __ESBMC_assert(__ESBMC_exists(&i, i > x || x == 4294967295u), "some larger");
  __ESBMC_assert(__ESBMC_forall(&i, i <= 4294967295u), "trivial");
  return 0;
}
