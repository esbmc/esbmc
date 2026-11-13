unsigned nondet_uint(void);
int main(void) {
  unsigned x = nondet_uint();
  __ESBMC_assert(x == x, "refl");
  return 0;
}
