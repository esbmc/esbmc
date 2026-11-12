unsigned nondet_uint(void);
int main(void) {
  unsigned c = nondet_uint(), d = nondet_uint();
  if (c) {
    if (d) goto L;
    c = 1;
  }
L:
  __ESBMC_assert(c != 2, "c");
  return 0;
}
