unsigned nondet_uint(void);
int main(void) {
  unsigned x = nondet_uint(), y = nondet_uint();
  __ESBMC_assume(y == 0 && x == 7);
  unsigned q = x / y;
  __ESBMC_assert(q == 0, "q");
  return 0;
}
