unsigned nondet_uint(void);
int main(void) {
  unsigned x = nondet_uint();
  __ESBMC_assert(__ESBMC_forall((void *)1, x == x), "q");
  return 0;
}
