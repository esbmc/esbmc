unsigned nondet_uint(void);
int main(void) {
  unsigned c = nondet_uint(), x = 0;
  if (c) {
  L:
    x = x + 1;
  }
  if (x < c) {
    __ESBMC_assert(x != 0, "nz");
    goto L;
  }
  return 0;
}
