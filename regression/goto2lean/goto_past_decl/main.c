unsigned nondet_uint(void);
int main(void) {
  unsigned c = nondet_uint();
  unsigned d = nondet_uint();
  if (d) goto L;
  goto L;
  unsigned x;
L:
  x = 0;
  if (c) x = 1;
  __ESBMC_assert(x == 0, "x is zero");
  return 0;
}
